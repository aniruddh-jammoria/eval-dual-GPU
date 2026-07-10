# 0001. Store model/run tags as explicit CSV columns, not id substrings

Date: 2026-07-10
Status: Accepted

## Context

The dashboard needed a way to know which models are MoE (for a badge) and, going forward, which runs used non-default settings (flash-attn off, mmap off, thinking off). The original MoE implementation inferred this by checking whether the model id contained `"moe"`, `"a3b"`, or `"a4b"` — no column stored it anywhere. That worked only because the current 8 model ids happen to contain a matching substring. It's fragile: a future model id could contain one of those substrings by coincidence, or a genuine runtime toggle (like thinking) can't be expressed as a static id substring at all, since it can vary per-request for the same underlying model.

## Decision

Model traits and run-time toggles are recorded as explicit fields, threaded end-to-end through the pipeline (Python dict → printed log line → `log2csv.py` regex → CSV column), the same way `gpu_config` already works. `generate_report.py` reads the explicit column when present, falling back to the old substring check only for historical CSVs that predate the column.

## Alternatives considered

- **Keep substring inference, extend the convention** (e.g. require all thinking-off ids to end in `-nothink`): rejected because it conflates model identity with configuration state, and doesn't generalize to axes that vary per-request rather than per-model.
- **Store tags only in `results.csv`, not the per-session metrics CSVs**: rejected because the dashboard reads exclusively from `results/metrics/*.csv`, not `results.csv` — tags absent from that path would just never reach the dashboard.

## Consequences

Adding a new tag/axis now means: add a field to the relevant Python dict (`MODELS[]` entry for static traits, a CLI flag for runtime toggles), print it into the log text, add a regex to `log2csv.py`, add a fallback-aware read in `generate_report.py`. More plumbing per axis than a one-line substring check, but each axis is then correct by construction instead of by naming discipline, and runtime toggles (which substrings can't represent at all) are supported the same way as static traits. Old CSVs written before a given tag existed keep working via the substring/default fallback rather than erroring or silently mis-tagging.
