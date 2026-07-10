# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- RAM metric: tracks peak resident memory (RSS) of the actual inference process (not whole-system RAM) during each benchmark run, sampled every 250ms via `psutil`. Surfaced as a new dashboard metric pill and in `bench`/`run-all` console output.
- `--flash-attn {on,off}` and `--mmap {on,off}` flags on `bench`/`run-all` to benchmark llama.cpp with flash attention or mmap disabled.
- Explicit `moe` field on each model entry, recorded per run — replaces guessing MoE-ness from the model id string, so the dashboard's MoE badge no longer depends on a naming convention.

### Fixed
- Fixed a log-parsing bug where using `--mtp` broke metrics-CSV parsing for that run (the log line's embedded space made the old regex unmatchable), so those runs' GPU config silently never reached the dashboard pipeline.

### Changed
- README: reframed the intro to state the single-GPU benchmark gap and the dual-5060-Ti cost/limits tradeoff up front instead of opening with a question; reordered Methodology before "How it works"; removed the "Key findings" section; documented the new RAM metric.
