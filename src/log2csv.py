"""
Parse metrics log files and output a CSV.

Usage:
  python log2csv.py                           # all results/logs/*.log → results/metrics/results.csv
  python log2csv.py path/to/file.log          # single file → stdout
  python log2csv.py results/logs/*.log -o out.csv
"""

import csv
import re
import sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent   # repo root (src/ lives under it)
LOGS_DIR    = ROOT / "results" / "logs"
METRICS_DIR = ROOT / "results" / "metrics"

FIELDS = [
    "log_file", "model", "tier", "backend", "gpu_config",
    "run_1_decode_tok_s", "run_1_prompt_tok_s", "run_1_ttft_s",
    "run_2_decode_tok_s", "run_2_prompt_tok_s", "run_2_ttft_s",
    "decode_tok_s", "prompt_tok_s", "ttft_s",
    "bw_gb_s", "bw_pct",
    "gpu0_gib", "gpu1_gib",
    "avg_watts", "peak_watts",
]

# ── Regex patterns ─────────────────────────────────────────────────────────────
RE_MODEL     = re.compile(r"^={3,}\s*$")
RE_MODEL_NAME= re.compile(r"^\s{2}(\S.*\S|\S)\s*$")
RE_TIER      = re.compile(r"── tier:\s*(\w+)")
RE_BLOCK_LC  = re.compile(r"\[llamacpp \| (\S+)\]")
RE_BLOCK_OL  = re.compile(r"\[ollama\]")
RE_RUN       = re.compile(
    r"run (\d+)/\d+\s*\.\.\.\s*([\d.]+) tok/s\s+\(prompt\s+([\d.]+) tok/s\s+TTFT\s+([\d.]+)s\)"
)
RE_DECODE    = re.compile(r"decode\s+([\d.]+)\s+tok/s")
RE_PROMPT    = re.compile(r"prompt\s+([\d.]+)\s+tok/s\s+TTFT\s+([\d.]+)s")
RE_BW        = re.compile(r"bw\s+([\d.]+)\s+GB/s\s+\(([\d.]+)%")
RE_GPU       = re.compile(r"GPU\s+(\d+)\s+([\d.]+)\s+GiB")
RE_POWER     = re.compile(r"power\s+([\d.]+)W avg\s+([\d.]+)W peak")


def parse_log(path: Path) -> list[dict]:
    rows = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    model      = ""
    tier       = ""
    in_sep     = False   # just saw a === line
    cur: dict | None = None

    def flush():
        if cur and cur.get("decode_tok_s") is not None:
            rows.append(cur)

    i = 0
    while i < len(lines):
        line = lines[i]

        # === separator: first one opens, second one closes
        if RE_MODEL.match(line):
            if in_sep:
                # closing separator — model name already captured
                in_sep = False
            else:
                flush()
                cur = None
                in_sep = True
            i += 1
            continue

        if in_sep:
            m = RE_MODEL_NAME.match(line)
            if m:
                model = m.group(1).strip()
            i += 1
            continue

        # tier
        m = RE_TIER.search(line)
        if m:
            tier = m.group(1)
            i += 1
            continue

        # block header — llamacpp
        m = RE_BLOCK_LC.search(line)
        if m:
            flush()
            cur = {f: None for f in FIELDS}
            cur["log_file"]   = path.name
            cur["model"]      = model
            cur["tier"]       = tier
            cur["backend"]    = "llamacpp"
            cur["gpu_config"] = m.group(1)
            i += 1
            continue

        # block header — ollama
        if RE_BLOCK_OL.search(line):
            flush()
            cur = {f: None for f in FIELDS}
            cur["log_file"]   = path.name
            cur["model"]      = model
            cur["tier"]       = tier
            cur["backend"]    = "ollama"
            cur["gpu_config"] = "auto"
            i += 1
            continue

        if cur is None:
            i += 1
            continue

        # run N/2
        m = RE_RUN.search(line)
        if m:
            n = int(m.group(1))
            prefix = f"run_{n}_"
            cur[prefix + "decode_tok_s"] = float(m.group(2))
            cur[prefix + "prompt_tok_s"] = float(m.group(3))
            cur[prefix + "ttft_s"]       = float(m.group(4))
            i += 1
            continue

        # avg decode
        m = RE_DECODE.search(line)
        if m and "prompt" not in line:
            cur["decode_tok_s"] = float(m.group(1))
            i += 1
            continue

        # avg prompt + ttft  (must come before bw/gpu checks)
        m = RE_PROMPT.search(line)
        if m and "run " not in line and "bw" not in line:
            cur["prompt_tok_s"] = float(m.group(1))
            cur["ttft_s"]       = float(m.group(2))
            i += 1
            continue

        # bandwidth
        m = RE_BW.search(line)
        if m:
            cur["bw_gb_s"] = float(m.group(1))
            cur["bw_pct"]  = float(m.group(2))
            i += 1
            continue

        # GPU N vram
        m = RE_GPU.search(line)
        if m:
            cur[f"gpu{m.group(1)}_gib"] = float(m.group(2))
            i += 1
            continue

        # GPU power
        m = RE_POWER.search(line)
        if m:
            cur["avg_watts"]  = float(m.group(1))
            cur["peak_watts"] = float(m.group(2))
            i += 1
            continue

        i += 1

    flush()
    return rows


def write_csv(path: Path, rows: list[dict]):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="*", type=Path,
                   help="Log file(s) to parse. Default: all results/logs/*.log")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Output CSV (only when a single log is given; "
                        "batch mode always writes one CSV per log file)")
    args = p.parse_args()

    files = args.logs or sorted(LOGS_DIR.glob("*.log"))
    if not files:
        sys.exit(f"No log files found in {LOGS_DIR}")

    # Single file + explicit -o → write to that path (or stdout if omitted)
    if len(files) == 1:
        rows = parse_log(files[0])
        print(f"  {files[0].name}: {len(rows)} rows", file=sys.stderr)
        out = args.out
        if out:
            write_csv(out, rows)
            print(f"  → {out}", file=sys.stderr)
        else:
            w = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        return

    # Batch mode: one CSV per log file in results/metrics/
    total = 0
    for f in files:
        rows = parse_log(f)
        if rows:
            out = METRICS_DIR / (f.stem + ".csv")
            write_csv(out, rows)
            print(f"  {f.name}: {len(rows)} rows → {out.name}", file=sys.stderr)
            total += len(rows)
        else:
            print(f"  {f.name}: 0 rows (skipped)", file=sys.stderr)
    print(f"\n  {total} rows written to {METRICS_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
