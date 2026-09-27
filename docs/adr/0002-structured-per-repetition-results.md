# 0002. Write per-repetition results as structured CSVs, not via log parsing

Date: 2026-09-27
Status: Accepted (amends the pipeline described in [0001](0001-model-tagging-explicit-columns.md))

## Context

Until 0.1.0, the dashboard's input was produced by printing results to the console log and
then regex-parsing that log back into `results/metrics/*.csv` (`log2csv.py`). ADR 0001 made
every new field travel that path: Python dict → printed line → regex → CSV column. It had
already broken once (the `--mtp` header bug silently dropped runs), and the paper needs
more than it can carry: ≥5 repetitions per cell with confidence intervals, per-GPU
telemetry that varies with GPU count, failure rows, and an environment manifest.

## Decision

`run.py` writes structured output directly:
- `results/runs/<session>.csv` — one row per repetition (and per failure), appended and
  flushed as each request finishes, so a crashed session keeps everything collected so far.
- `results/runs/<session>.env.json` — hardware/software manifest.
- `results/metrics/<session>.csv` — per-cell aggregates (mean, 95% CI, n, failures) computed
  by `summarize.py` from the runs rows, in the legacy column schema plus new columns, so
  `generate_report.py` keeps working unchanged.

The console log stays human-readable only. New logs start with `# schema: v2` and
`log2csv.py` skips them, so re-parsing legacy logs can't overwrite the new metrics CSVs.

## Alternatives considered

- **Extend the log format and regexes**: rejected — each repetition, GPU and telemetry field
  would need a new regex, and silent parse failures are exactly the failure mode to avoid.
- **SQLite/Parquet instead of CSV**: rejected for now — CSV is diffable in git, readable by
  reviewers without tooling, and small enough at this scale.

## Consequences

Adding a field is now: set it on the row dict in `run.py` (and add to `RUN_FIELDS`). ADR
0001's principle — traits and toggles as explicit columns, not id substrings — still holds;
only the transport changed. Legacy 0.1.0 data remains readable through `log2csv.py` and the
old metrics CSVs. `results/results.csv` is still appended for `analyze.py` but is no longer
the source of truth.
