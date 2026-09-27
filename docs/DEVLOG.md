# Development Log

Newest entries first. Major decisions live in [adr/](adr/).

## 2026-09-27 — Protocol overhaul for the paper (v0.2.0), paper outline, first releases

**Type:** feature / analysis
**Context:** The repo is becoming a portfolio piece and the basis of a self-published paper (Zenodo, then arXiv cs.DC/cs.PF). A methodology review found the 0.1.0 data couldn't support citable claims: 2 repetitions with no variance, run-2 prefill served from warm KV cache, output length varying per model, a bandwidth metric that exceeded physical peak for MoE, Ollama mislabelled as GPU0-only, failed runs silently missing, and no record of software versions.

**What was done:**
- Wrote [paper/OUTLINE.md](paper/OUTLINE.md): research questions, section plan, experiments A–K (slot swap, link-bandwidth sweep, cloud 5090 and symmetric dual-GPU references, Linux control), methodology fixes M1–M11, submission checklist.
- Implemented M1–M5, M7–M9, M11 (see CHANGELOG 0.2.0). Structural change: results are written directly as per-repetition CSVs instead of regex-parsed from logs — see [adr/0002](adr/0002-structured-per-repetition-results.md). New modules: `config.py` (paths + `models.toml` registry), `telemetry.py`, `model_info.py` (GGUF header introspection via the `gguf` package), `summarize.py`.
- M6 turned out mostly solvable in software: GPU0 now reports NVML power (the old "NotSupported" comment is stale — likely fixed by a driver update), and both cards expose a hardware energy counter, so per-request energy is exact.
- Smoke-tested on the rig (Qwen3.5 9B Q4, n=2) and deliberately triggered the known Gemma 12B tensor-split crash to verify failure recording; test outputs deleted.
- Tagged v0.1.0 retroactively on the 2026-07-10 commit and v0.2.0 for this work; added `CITATION.cff` and an "⚠ Affects measurements" changelog convention.

**Findings worth keeping:**
- PCIe links measured under load: GPU0 = **5.0 x8** (the card is x8-electrical), GPU1 = **4.0 x2**. The README's "4.0 x16" was wrong.
- Tensor split during decode moves only ~170–350 MB/s per GPU over PCIe (<10% of the x2 link), while tensor-split *prefill* is ~1.6× slower than single-GPU — consistent with "decode is latency-bound, prefill is link-bound".
- Gemma 12B tensor split crashes in llama.cpp b9858 (`GGML_ASSERT` in `ggml-backend-meta.cpp`) — a reason to update llama.cpp once before collecting final data, then freeze it.

**Outcome:** Harness ready for final data collection pending a llama.cpp update. Not yet tested on Linux (M11). Open: LICENSE choice (needed for Zenodo), figure scripts (M10), final model list.

## 2026-07-10 — RAM tracking, flash-attn/mmap config axes, explicit MoE tagging

**Type:** feature
**Context:** User wanted to track system RAM usage during benchmarks (to see CPU-spill impact directly), and to test flash-attention and mmap on/off as new configuration axes, following up on items already listed in `Ideas.md`.

**What was done:**
- Added a `RamSampler` mirroring the existing `VramSampler`/`PowerSampler` pattern. Initially built as system-wide `psutil.virtual_memory().used`, but the user correctly pushed back that this isn't attributable to the model — switched to sampling the actual inference process's RSS instead: the exact PID for llama.cpp (already spawned via `Popen` in the `llama_server` context manager, which now yields the process handle), and a heaviest-match-by-name scan for Ollama (a persistent service that may fan work to a model-runner subprocess, so no fixed PID to hold onto).
- Added `--flash-attn {on,off}` and `--mmap {on,off}` as CLI flags on `bench`/`run-all`, threaded through to the actual llama.cpp server flags. Modeled on the existing `--mtp` precedent (a scalar flag recorded per-run, not multiplied combinatorially against GPU configs) rather than expanding `GPU_CONFIGS` — the 5-column dashboard grid can't absorb more axes without a redesign, so these are for targeted comparisons via `run.py results`, not full dashboard columns yet.
- While wiring the flash-attn/mmap tags into the block-header log line, found and fixed a **pre-existing bug**: the old `[llamacpp | {cfg}{mtp_tag}]` header format was unparseable by `log2csv.py`'s regex whenever `--mtp` was used, because the tag's embedded space broke the `(\S+)\]` capture. Runs using `--mtp` silently never got their `gpu_config` into the metrics CSVs feeding the dashboard. Replaced the format with an explicit `cfg=X fa=Y mmap=Z mtp=N` key=value header, parsed by one robust regex — verified against synthetic log lines including the mtp case.
- Replaced substring-based MoE detection (`is_moe(mid)` guessing from `"moe"/"a3b"/"a4b"` in the model id) with an explicit `"moe": bool` field on each `MODELS[]` entry, threaded through the log text (`moe: True/False` printed once per model, parsed into a state variable analogous to how `model`/`tier` already thread through `log2csv.py`) into a real CSV column. `generate_report.py` prefers the explicit column but falls back to substring inference when it's absent, so historical CSVs written before this change keep rendering correctly.
- Scaffolded (but did not wire) a `thinking:off` tag: added `is_thinking_off(mid)` and a dashboard badge on the `-nothink` id-suffix convention, mirroring MoE's old approach. Deliberately did not implement the actual request-time toggle — thinking on/off is controlled per-backend/per-model (Ollama's `think` param vs. a llama.cpp chat-template kwarg vs. prompt convention) and guessing wrong would silently produce mislabeled benchmark data. Held pending user confirmation of the mechanism per model.

**Outcome:** RAM, flash-attn/mmap, and MoE tagging are implemented and verified (compile checks, synthetic log-parsing tests, and a real `generate_report.py` run against the existing CSVs, which correctly fall back to old behavior since they predate all three new columns). Not yet tested against a live benchmark run (no GPU/Ollama/llama-server available in this environment) — first real `bench` invocation after this change should be checked against expectations. Open follow-up: nail down the thinking on/off toggle mechanism per model/backend before implementing it for real. See [adr/0001-model-tagging-explicit-columns.md](adr/0001-model-tagging-explicit-columns.md) for the tagging-approach decision.
