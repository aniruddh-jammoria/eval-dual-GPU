"""
Aggregate per-repetition rows (results/runs/*.csv) into per-cell statistics.

Each cell = (model, tier, backend, gpu_config, flash_attn, mmap, mtp_n, ignore_eos).
Reports n, mean, sample SD, 95% CI half-width (Student t), CV, and failure counts.

Usage:
  python src/summarize.py                     print summary of all sessions, write results/summary.csv
  python src/summarize.py --session <stem>    only one session
  python src/summarize.py --rebuild-metrics   regenerate results/metrics/<session>.csv for every
                                              runs file (e.g. after a crashed session)
"""

import argparse
import re
import csv
import statistics
import sys
from collections import defaultdict

from config import METRICS_DIR, RESULTS_DIR, RUNS_DIR

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SUMMARY_CSV = RESULTS_DIR / "summary.csv"

CELL_KEYS = ["model_id", "tier", "backend", "gpu_config",
             "flash_attn", "mmap", "mtp_n", "ignore_eos", "ubatch"]

STAT_METRICS = ["decode_tok_s", "prefill_tok_s", "ttft_s", "prompt_n", "n_generated",
                "wall_s", "energy_j", "decode_watts", "decode_j_per_tok",
                "avg_watts", "peak_watts", "peak_rss_mib", "bw_gb_s", "bw_pct"]

# two-sided 95% Student t critical values, df = 1..30
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]

def t95(df):
    return _T95[df - 1] if 1 <= df <= len(_T95) else 1.960

def describe(vals):
    """n, mean, sd, ci95 half-width, cv — sd/ci None when n < 2."""
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": None, "cv": None}
    mean = statistics.fmean(vals)
    if n < 2:
        return {"n": 1, "mean": mean, "sd": None, "ci95": None, "cv": None}
    sd = statistics.stdev(vals)
    return {"n": n, "mean": mean, "sd": sd,
            "ci95": t95(n - 1) * sd / n ** 0.5,
            "cv": sd / mean if mean else None}

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def load_runs(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8", newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows

def aggregate(rows):
    """→ list of dicts: cell keys + descriptive fields + '<metric>_<stat>' columns."""
    cells = defaultdict(list)
    for r in rows:
        cells[tuple(r.get(k, "") for k in CELL_KEYS)].append(r)

    out = []
    for key, rs in cells.items():
        ok   = [r for r in rs if str(r.get("ok")) == "True"]     # bool in memory, str from CSV
        fail = [r for r in rs if str(r.get("ok")) != "True"]
        first = (ok or rs)[0]
        agg = dict(zip(CELL_KEYS, key))
        agg |= {
            "model_name":    first.get("model_name", ""),
            "moe":           first.get("moe", ""),
            "session":       rs[-1].get("session", ""),
            "n_ok":          len(ok),
            "n_failed":      len(fail),
            "errors":        " | ".join(sorted({(r.get("error") or "")[:120] for r in fail})),
            "observed_gpus": ";".join(sorted({r.get("observed_gpus", "") for r in ok})),
            "throttled":     any(r.get("throttle") for r in ok),
            "max_tokens":    first.get("max_tokens", ""),
            "hit_max_frac":  (sum(1 for r in ok if _f(r.get("n_generated")) == _f(r.get("max_tokens")))
                              / len(ok)) if ok else None,
        }
        for m in STAT_METRICS:
            d = describe([_f(r.get(m)) for r in ok])
            agg |= {f"{m}_mean": d["mean"], f"{m}_sd": d["sd"],
                    f"{m}_ci95": d["ci95"], f"{m}_cv": d["cv"]}
        # per-GPU peak VRAM (mean over reps) — columns vary with GPU count
        gpu_cols = sorted({k for r in ok for k in r if k.endswith("_vram_mib") and k.startswith("gpu")})
        for c in gpu_cols:
            agg[f"{c}_mean"] = describe([_f(r.get(c)) for r in ok])["mean"]
        out.append(agg)
    return out

# ── Dashboard-compatible metrics CSV (legacy schema + CI columns) ─────────────
LEGACY_FIELDS = [
    "log_file", "model", "moe", "tier", "backend", "gpu_config",
    "flash_attn", "mmap", "ubatch",
    "decode_tok_s", "prompt_tok_s", "ttft_s",
    "bw_gb_s", "bw_pct",
    "gpu0_gib", "gpu1_gib",
    "avg_watts", "peak_watts",
    "peak_ram_gib", "ram_total_gib",
    # v2 additions
    "schema", "n", "n_failed", "errors", "observed_gpus", "throttled",
    "decode_sd", "decode_ci95", "prompt_ci95", "ttft_ci95",
    "decode_j_per_tok", "energy_j", "hit_max_frac",
]

def metrics_rows(agg_rows, session):
    out = []
    for a in agg_rows:
        gib = lambda c: (a.get(c) / 1024) if a.get(c) is not None else None
        rss = a.get("peak_rss_mib_mean")
        out.append({
            "log_file":      session + ".log",
            "model":         f"{a['model_id']}  —  {a['model_name']}",
            "moe":           a["moe"],
            "tier":          a["tier"],
            "backend":       a["backend"],
            "gpu_config":    a["gpu_config"],
            "flash_attn":    a["flash_attn"],
            "mmap":          a["mmap"],
            "ubatch":        a["ubatch"],
            "decode_tok_s":  a["decode_tok_s_mean"],
            "prompt_tok_s":  a["prefill_tok_s_mean"],
            "ttft_s":        a["ttft_s_mean"],
            "bw_gb_s":       a["bw_gb_s_mean"],
            "bw_pct":        a["bw_pct_mean"],
            "gpu0_gib":      gib("gpu0_vram_mib_mean"),
            "gpu1_gib":      gib("gpu1_vram_mib_mean"),
            "avg_watts":     a["avg_watts_mean"],
            "peak_watts":    a["peak_watts_mean"],
            "peak_ram_gib":  rss / 1024 if rss else None,
            "ram_total_gib": None,
            "schema":        "v2",
            "n":             a["n_ok"],
            "n_failed":      a["n_failed"],
            "errors":        a["errors"],
            "observed_gpus": a["observed_gpus"],
            "throttled":     a["throttled"],
            "decode_sd":     a["decode_tok_s_sd"],
            "decode_ci95":   a["decode_tok_s_ci95"],
            "prompt_ci95":   a["prefill_tok_s_ci95"],
            "ttft_ci95":     a["ttft_s_ci95"],
            "decode_j_per_tok": a["decode_j_per_tok_mean"],
            "energy_j":      a["energy_j_mean"],
            "hit_max_frac":  a["hit_max_frac"],
        })
    return out

def write_metrics(session, runs_rows):
    """Write results/metrics/<session>.csv; failed-only cells included (decode empty)."""
    rows = metrics_rows(aggregate(runs_rows), session)
    if not rows:
        return None
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{session}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEGACY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path

# ── CLI ───────────────────────────────────────────────────────────────────────
def _fmt(mean, ci, digits=1):
    if mean is None:
        return "—"
    return f"{mean:.{digits}f} ± {ci:.{digits}f}" if ci is not None else f"{mean:.{digits}f}"

def print_table(agg_rows):
    # natural sort so sweep tiers order pp128 < pp1024 rather than as strings
    nat = lambda v: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(v))]
    agg_rows = sorted(agg_rows, key=lambda a: [nat(a[k]) for k in CELL_KEYS])
    print(f"\n  {'model':<22} {'tier':<8} {'backend':<9} {'gpu':<12} {'n':>3}"
          f" {'decode tok/s':>15} {'CV':>6} {'prefill tok/s':>15} {'TTFT s':>13}"
          f" {'J/tok':>7} {'gpus':>5}  notes")
    print("  " + "─" * 132)
    for a in agg_rows:
        notes = []
        if a["n_failed"]:
            notes.append(f"{a['n_failed']} failed")
        if a["throttled"]:
            notes.append("throttled")
        if a["ubatch"] not in ("", "512", 512, None):
            notes.append(f"ub={a['ubatch']}")
        if a["flash_attn"] not in ("", "on") or a["mmap"] not in ("", "on"):
            notes.append(f"fa={a['flash_attn']} mmap={a['mmap']}")
        cv = a["decode_tok_s_cv"]
        jt = a["decode_j_per_tok_mean"]
        print(f"  {a['model_id']:<22} {a['tier']:<8} {a['backend']:<9} {a['gpu_config']:<12}"
              f" {a['n_ok']:>3} {_fmt(a['decode_tok_s_mean'], a['decode_tok_s_ci95']):>15}"
              f" {(f'{cv*100:.1f}%' if cv is not None else '—'):>6}"
              f" {_fmt(a['prefill_tok_s_mean'], a['prefill_tok_s_ci95'], 0):>15}"
              f" {_fmt(a['ttft_s_mean'], a['ttft_s_ci95'], 3):>13}"
              f" {(f'{jt:.2f}' if jt else '—'):>7} {a['observed_gpus']:>5}  {', '.join(notes)}")
    print(f"\n  {len(agg_rows)} cells  |  mean ± 95% CI (Student t)  |  CV = SD / mean of decode tok/s\n")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", help="runs file stem to summarise (default: all)")
    p.add_argument("--rebuild-metrics", action="store_true",
                   help="rewrite results/metrics/<session>.csv from each runs file")
    args = p.parse_args()

    paths = sorted(RUNS_DIR.glob(f"{args.session or '*'}.csv"))
    if not paths:
        sys.exit(f"No runs files in {RUNS_DIR}" + (f" matching {args.session}" if args.session else ""))

    if args.rebuild_metrics:
        for pth in paths:
            out = write_metrics(pth.stem, load_runs([pth]))
            print(f"  {pth.name} → {out}")
        return

    agg = aggregate(load_runs(paths))
    print_table(agg)
    fields = list(dict.fromkeys(k for a in agg for k in a))
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(agg)
    print(f"  summary → {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
