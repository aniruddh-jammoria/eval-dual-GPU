# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Versioning for a benchmark.** A change that alters the numbers the harness produces
is a minor bump before 1.0 and a major bump after. Such entries are marked
**⚠ Affects measurements** — results collected on either side of one are not directly
comparable. Every session records the harness version (`git describe`) in its
`results/runs/<session>.env.json`, so any result can be traced to the release that produced it.

## [Unreleased]

## [0.2.0] - 2026-09-27

Measurement protocol overhaul in preparation for the paper. Data collected with 0.1.0
should be treated as preliminary.

### Added
- **⚠ Affects measurements.** Publication-grade measurement protocol: 5 timed repetitions per cell by default (`--repeats`), one untimed warm-up per tier, seeded-random order of configurations and tiers (`--seed`, `--no-shuffle`), cooldown and GPU-temperature gate before each configuration (`--cooldown`, `--max-start-temp`).
- Per-repetition results in `results/runs/<session>.csv` with full telemetry per GPU: VRAM, power, temperature, SM clock, PCIe link generation/width under load, PCIe RX/TX throughput, throttle reasons, and exact energy from the hardware energy counter.
- Environment manifest per session (`results/runs/<session>.env.json`): driver, CUDA, GPU UUID/VBIOS/power limit/PCIe capability, llama.cpp and Ollama versions, harness version and commit, CLI arguments. Also `python src/run.py env`.
- `python src/summarize.py`: mean ± 95% confidence interval, SD and CV per cell; `--rebuild-metrics` regenerates dashboard CSVs from raw runs.
- Model registry in `models.toml` — adding a model is one block in one file, and the dashboard picks it up automatically. Architecture (layers, hidden size, experts, bytes read per token) is read from the GGUF header (`python src/model_info.py`).
- Machine paths in `config.toml` or environment variables (`GGUF_DIR`, `LLAMACPP_BIN`, …); Ollama is optional and skipped when absent. Runs on Linux as well as Windows.
- `run-all --models …` to benchmark a subset.
- Decode energy per token (J/token) and the GPUs a model actually ran on (`observed_gpus`), recorded per run.
- `CITATION.cff` so the repository can be cited.

### Changed
- Results are written directly as structured CSVs instead of being parsed back out of the console log; `log2csv.py` now only handles pre-0.2.0 logs.

### Fixed
- **⚠ Affects measurements.** Prefill/TTFT no longer measured from a warm KV cache: every request gets a random nonce prefix and llama.cpp gets `cache_prompt: false` (verified via `cached_n = 0`).
- **⚠ Affects measurements.** Decode speed no longer depends on how long each model chooses to answer: llama.cpp requests use `ignore_eos` to generate exactly `max_tokens`.
- **⚠ Affects measurements.** Bandwidth metric now uses bytes actually read per token instead of GGUF file size, so MoE models no longer report more than the hardware peak.
- Failed runs (server crashes, request errors) are recorded and reported instead of silently missing from the dashboard.
- Ollama was labelled "GPU0" but places larger models across both GPUs; placement is now measured per run.
- Hardware table and dashboard: GPU0's link is PCIe 5.0 x8 (measured under load; the RTX 5060 Ti is an x8 card), not PCIe 4.0 x16.

## [0.1.0] - 2026-07-10

First tagged version: benchmark CLI for 8 GGUF models across Ollama and four llama.cpp GPU
configurations (single GPU0, single GPU1, layer split, tensor split), four prompt tiers,
and a static GitHub Pages dashboard. Its data (2026-07 sessions) is preliminary: 2
repetitions, warm-cache prefill on run 2, variable output length.

### Added
- RAM metric: tracks peak resident memory (RSS) of the actual inference process (not whole-system RAM) during each benchmark run, sampled every 250ms via `psutil`. Surfaced as a new dashboard metric pill and in `bench`/`run-all` console output.
- `--flash-attn {on,off}` and `--mmap {on,off}` flags on `bench`/`run-all` to benchmark llama.cpp with flash attention or mmap disabled.
- Explicit `moe` field on each model entry, recorded per run — replaces guessing MoE-ness from the model id string, so the dashboard's MoE badge no longer depends on a naming convention.

### Fixed
- Fixed a log-parsing bug where using `--mtp` broke metrics-CSV parsing for that run (the log line's embedded space made the old regex unmatchable), so those runs' GPU config silently never reached the dashboard pipeline.

### Changed
- README: reframed the intro to state the single-GPU benchmark gap and the dual-5060-Ti cost/limits tradeoff up front instead of opening with a question; reordered Methodology before "How it works"; removed the "Key findings" section; documented the new RAM metric.

[Unreleased]: https://github.com/aniruddh-jammoria/eval-dual-GPU/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/aniruddh-jammoria/eval-dual-GPU/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/aniruddh-jammoria/eval-dual-GPU/releases/tag/v0.1.0
