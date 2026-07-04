"""
Analyse benchmark results from results/results.csv.

Usage:
  python analyze.py                   full report (tables + charts)
  python analyze.py --table           tables only
  python analyze.py --charts          charts only
  python analyze.py --tier chat       filter to one prompt tier
"""

import argparse
import ast
import csv
import sys
from pathlib import Path

RESULTS_CSV = Path(__file__).parent / "results" / "results.csv"
RESULTS_DIR = Path(__file__).parent / "results"
TIER_ORDER  = ["chat", "rag", "longdoc", "code"]


def load(tier_filter=None):
    if not RESULTS_CSV.exists():
        sys.exit("No results yet — run: python run.py run-all")
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    ok   = [r for r in rows if r.get("ok") == "True"]
    if not ok:
        sys.exit("No successful results yet.")
    if tier_filter:
        ok = [r for r in ok if r.get("prompt_tier") == tier_filter]
        if not ok:
            sys.exit(f"No results for tier '{tier_filter}'.")
    for r in ok:
        for col in ["decode_tok_per_s", "prompt_tok_per_s", "ttft_s",
                    "load_time_s", "bandwidth_gb_s", "bandwidth_pct", "gguf_size_gb",
                    "peak_watts", "avg_watts"]:
            try:   r[col] = float(r[col])
            except Exception: r[col] = 0.0
    return ok


# ── Tables ────────────────────────────────────────────────────────────────────
def table_decode(rows):
    tiers = [t for t in TIER_ORDER if any(r.get("prompt_tier") == t for r in rows)]
    for tier in tiers:
        sub = [r for r in rows if r.get("prompt_tier") == tier]
        print(f"\n{'═'*88}")
        print(f"  DECODE SPEED (tok/s)  — tier: {tier}")
        print(f"{'═'*88}")
        print(f"  {'model':<24} {'backend':<10} {'gpu':<9}"
              f" {'decode t/s':>10} {'prompt t/s':>11} {'TTFT s':>7} {'bw%':>6}")
        print("  " + "─"*83)
        sub.sort(key=lambda r: (r.get("id",""), r.get("backend",""), r.get("gpu_config","")))
        for r in sub:
            bw = f"{r['bandwidth_pct']:>5.1f}%" if r["bandwidth_pct"] else "—"
            print(f"  {r.get('id','?'):<24} {r.get('backend','?'):<10}"
                  f" {r.get('gpu_config','?'):<9}"
                  f" {r['decode_tok_per_s']:>10.1f}"
                  f" {r['prompt_tok_per_s']:>11.0f}"
                  f" {r['ttft_s']:>7.3f}"
                  f" {bw:>6}")


def table_gpu_comparison(rows):
    lc = [r for r in rows if r.get("backend") == "llamacpp"]
    if not lc:
        return
    tiers = [t for t in TIER_ORDER if any(r.get("prompt_tier") == t for r in lc)]
    for tier in tiers:
        sub = [r for r in lc if r.get("prompt_tier") == tier]
        by_model = {}
        for r in sub:
            by_model.setdefault(r["id"], {})[r.get("gpu_config")] = r

        print(f"\n{'═'*78}")
        print(f"  GPU CONFIG COMPARISON — llama.cpp — tier: {tier}")
        print(f"  dual_21 = 2/3 layers on GPU0 (x16 slot), 1/3 on GPU1 (x2 slot)")
        print(f"{'═'*78}")
        print(f"  {'model':<24} {'single0':>9} {'single1':>9} {'dual':>9}"
              f" {'dual_21':>9} {'dual vs s0':>11}")
        print("  " + "─"*73)
        for id_, cfgs in sorted(by_model.items()):
            def tok(c): return f"{cfgs[c]['decode_tok_per_s']:>9.1f}" if c in cfgs else f"{'—':>9}"
            diff = ""
            if "single0" in cfgs and "dual" in cfgs:
                d = (cfgs["dual"]["decode_tok_per_s"] - cfgs["single0"]["decode_tok_per_s"])
                diff = f"  {d/cfgs['single0']['decode_tok_per_s']*100:>+.1f}%"
            print(f"  {id_:<24}{tok('single0')}{tok('single1')}{tok('dual')}"
                  f"{tok('dual_21')}{diff:>11}")


def table_backend_comparison(rows):
    tiers = [t for t in TIER_ORDER if any(r.get("prompt_tier") == t for r in rows)]
    for tier in tiers:
        sub = [r for r in rows if r.get("prompt_tier") == tier]
        by_model = {}
        for r in sub:
            by_model.setdefault(r["id"], {})[r.get("backend")] = r

        print(f"\n{'═'*65}")
        print(f"  OLLAMA vs LLAMA.CPP — tier: {tier}  (llama.cpp = best dual config)")
        print(f"{'═'*65}")
        print(f"  {'model':<24} {'ollama t/s':>10} {'llamacpp t/s':>13} {'delta':>8}")
        print("  " + "─"*58)
        found = False
        for id_, backends in sorted(by_model.items()):
            o  = backends.get("ollama")
            lc_rows = [r for r in sub if r["id"] == id_ and r.get("backend") == "llamacpp"]
            lc = (next((r for r in lc_rows if r.get("gpu_config") == "dual_21"), None) or
                  next((r for r in lc_rows if r.get("gpu_config") == "dual"), None) or
                  next((r for r in lc_rows if r.get("gpu_config") == "single0"), None))
            if not o or not lc:
                continue
            found = True
            diff = (lc["decode_tok_per_s"] - o["decode_tok_per_s"]) / o["decode_tok_per_s"] * 100
            print(f"  {id_:<24} {o['decode_tok_per_s']:>10.1f}"
                  f" {lc['decode_tok_per_s']:>13.1f}  {diff:>+.1f}%")
        if not found:
            print("  (need results from both backends)")


def table_power(rows):
    has_power = any(r.get("avg_watts", 0) > 0 for r in rows)
    if not has_power:
        return
    chat = [r for r in rows if r.get("prompt_tier") == "chat"] or rows
    print(f"\n{'═'*72}")
    print("  POWER (GPU-only, decode phase)  — chat tier")
    print(f"{'═'*72}")
    print(f"  {'model':<24} {'backend':<10} {'gpu':<9}"
          f" {'avg W':>7} {'peak W':>8} {'W/tok/s':>9}")
    print("  " + "─"*67)
    for r in sorted(chat, key=lambda r: (r.get("id",""), r.get("backend",""))):
        if not r.get("avg_watts"):
            continue
        efficiency = r["avg_watts"] / max(r["decode_tok_per_s"], 0.001)
        print(f"  {r.get('id','?'):<24} {r.get('backend','?'):<10}"
              f" {r.get('gpu_config','?'):<9}"
              f" {r['avg_watts']:>7.1f} {r['peak_watts']:>8.1f} {efficiency:>9.2f}")


def table_vram(rows):
    dual = [r for r in rows if r.get("gpu_config") in ("dual", "dual_21")]
    if not dual:
        return
    print(f"\n{'═'*62}")
    print("  VRAM USAGE  (dual-GPU runs, chat tier)")
    print(f"{'═'*62}")
    print(f"  {'model':<24} {'cfg':<8} {'backend':<10} {'GPU0':>7} {'GPU1':>7} {'total':>7}")
    print("  " + "─"*58)
    chat = [r for r in dual if r.get("prompt_tier") == "chat"] or dual
    for r in sorted(chat, key=lambda r: r.get("id","")):
        try:
            vram = r.get("peak_vram_mib")
            if isinstance(vram, str):
                vram = ast.literal_eval(vram)
            g0 = vram.get(0, vram.get("0", 0)) / 1024
            g1 = vram.get(1, vram.get("1", 0)) / 1024
            print(f"  {r.get('id','?'):<24} {r.get('gpu_config','?'):<8}"
                  f" {r.get('backend','?'):<10} {g0:>6.1f}G {g1:>6.1f}G {g0+g1:>6.1f}G")
        except Exception:
            pass


# ── Charts ────────────────────────────────────────────────────────────────────
def make_charts(rows):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping charts.  pip install matplotlib")
        return

    COLORS = {
        "ollama / auto":       "#4A90D9",
        "llamacpp / single0":  "#E8A838",
        "llamacpp / single1":  "#F4C46A",
        "llamacpp / dual":     "#5CB85C",
        "llamacpp / dual_21":  "#2E7D32",
    }

    models  = sorted(set(r["id"] for r in rows))
    tiers   = [t for t in TIER_ORDER if any(r.get("prompt_tier") == t for r in rows)]
    labels  = [l for l in COLORS if l in {f"{r.get('backend')} / {r.get('gpu_config')}" for r in rows}]

    fig, axes = plt.subplots(1, len(tiers), figsize=(7*len(tiers), 5), sharey=True)
    if len(tiers) == 1:
        axes = [axes]

    for ax, tier in zip(axes, tiers):
        sub = [r for r in rows if r.get("prompt_tier") == tier]
        x   = np.arange(len(models))
        w   = 0.8 / max(len(labels), 1)
        for i, lbl in enumerate(labels):
            backend, gpu = lbl.split(" / ", 1)
            vals = []
            for mid in models:
                match = [r["decode_tok_per_s"] for r in sub
                         if r["id"] == mid and r.get("backend") == backend
                         and r.get("gpu_config") == gpu]
                vals.append(match[0] if match else 0)
            offset = (i - len(labels)/2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=lbl, color=COLORS[lbl], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=8)
        ax.set_title(f"tier: {tier}")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("tokens / sec")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Decode speed — model × backend × GPU config", y=1.01)
    plt.tight_layout()
    out = RESULTS_DIR / "decode_speed.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  chart saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Analyse dual-GPU eval results",
                                epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table",  action="store_true")
    p.add_argument("--charts", action="store_true")
    p.add_argument("--tier",   choices=TIER_ORDER, help="Filter to one prompt tier")
    args = p.parse_args()

    show_tables = args.table  or (not args.table  and not args.charts)
    show_charts = args.charts or (not args.table  and not args.charts)

    rows = load(tier_filter=args.tier)
    tier_label = f" (tier: {args.tier})" if args.tier else ""
    print(f"  {len(rows)} successful results{tier_label}")

    if show_tables:
        table_decode(rows)
        table_backend_comparison(rows)
        table_gpu_comparison(rows)
        table_power(rows)
        table_vram(rows)

    if show_charts:
        make_charts(rows)


if __name__ == "__main__":
    main()
