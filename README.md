# Dual RTX 5060 Ti 16 GB — LLM Inference Benchmark

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![CLI](https://img.shields.io/badge/type-CLI-informational)

Most LLM benchmarks available online are for single-GPU setups. The dual 5060 Ti is a
cheap way to get 32 GB of VRAM — two cards run **$1,000–1,200 combined, roughly 70–75%
cheaper** than a single-GPU Blackwell alternative like the RTX 5090 (~$4,000) — but is
often limited by factors like per-card VRAM capacity (16 GB, forcing a split across two
devices) and PCIe bandwidth to the second card (~4 GB/s over a chipset x2 slot). This
repo benchmarks 8 models across 2 LLM inference backends (Ollama and llama.cpp) and 5
GPU configurations, measuring decode speed, prefill speed, latency, memory bandwidth,
and power draw for each combination.

**[→ Live dashboard](https://aniruddh-jammoria.github.io/eval-dual-GPU/)**

---

## Hardware

| | GPU0 | GPU1 |
|---|---|---|
| Card | MSI VENTUS 2X OC | ASUS DUAL OC |
| Link (measured under load) | PCIe 5.0 x8 (CPU lanes) — ~32 GB/s | PCIe 4.0 x2 (B650 chipset) — ~4 GB/s |
| VRAM | 16 GB GDDR7 | 16 GB GDDR7 |
| Bandwidth | 672 GB/s | 672 GB/s |

**Platform:** AMD Ryzen 9 7900 · 32 GB DDR5 · MSI MAG B650 TOMAHAWK WIFI · Windows 11
**Software:** Ollama · llama.cpp b9858 (CUDA 13.3, Blackwell sm_120a)

The RTX 5060 Ti is an x8-electrical card, so GPU0 runs x8 even in an x16 slot. Link
generation and width are recorded per run (`gpuN_pcie_gen`, `gpuN_pcie_width`) — idle GPUs
downclock the link, so only under-load values are meaningful.

---

## Models tested

| Model | GGUF | Type |
|---|---:|---|
| Gemma 4 12B IT QAT (Q4_0) | 7.0 GB | Dense |
| Qwen3.5 9B (Q4\_K\_M) | 5.7 GB | Dense |
| Qwen3.5 9B (Q8\_0) | 9.5 GB | Dense |
| Qwen3.6 27B (Q4\_K\_M) | 16.8 GB | Dense |
| Qwen3.6 27B (Q6\_K) | 22.5 GB | Dense |
| Gemma 4 26B A4B IT MoE UD (Q4\_K\_M) | 16.9 GB | MoE |
| Qwen3.6 35B A3B MoE UD (Q4\_K\_M) | 22.1 GB | MoE |
| Gemma 4 31B IT UD (Q5\_K\_XL) | 21.9 GB | Dense |

---

## Configurations benchmarked

| Label | Backend | GPUs active | Notes |
|---|---|---|---|
| **Ollama** | Ollama | auto | Ollama chooses placement — GPU0 for models that fit, both GPUs for larger ones (recorded per run as `observed_gpus`) |
| **LC GPU0** | llama.cpp | GPU0 only | PCIe 5.0 x8, CPU lanes |
| **LC GPU1** | llama.cpp | GPU1 only | PCIe 4.0 x2, via chipset |
| **LC Dual** | llama.cpp | Both | Layer split — GPUs run sequentially per layer |
| **LC Dual ⊗** | llama.cpp | Both | Tensor split — GPUs run in parallel per layer |

---

## Metrics

| Metric | What it measures |
|---|---|
| **Decode tok/s** | Tokens generated per second — primary inference speed metric |
| **Prefill tok/s** | Prompt tokens processed per second — drives time-to-first-token |
| **TTFT (s)** | Time-to-first-token — latency before output starts |
| **Bandwidth (GB/s)** | Effective memory bandwidth: `bytes_read_per_token × decode_tok_s`, where bytes read per token come from the GGUF header (routed experts scaled by active/total) |
| **GPU Power (W)** | Average combined GPU power draw during inference |
| **Energy (J/token)** | Decode-phase GPU power ÷ decode tok/s; per-request energy from the NVML hardware energy counter |
| **RAM (GiB)** | Peak resident memory (RSS) of the inference process — spikes when VRAM overflow spills to system RAM |

Each cell is reported as mean ± 95% CI over 5 repetitions (default). Decode tok/s uses the backend's internal timing, not wall clock.

---

## Methodology

### Prompt tiers

Each model is benchmarked on four prompt types to test different context lengths and workloads:

| Tier | Input tokens | Max output | Prompt |
|---|---:|---:|---|
| `chat` | ~540 | 512 | Dracula Ch. I (400 words) — summarise first impressions |
| `rag` | ~2 000 | 1 024 | Dracula Ch. I (1 600 words) — list all warnings/dangers |
| `longdoc` | ~4 000 | 1 024 | Dracula Ch. I (3 200 words) — atmospheric analysis |
| `code` | ~155 | 1 024 | Implement `RateLimiter` class — sliding-window, thread-safe |

### Measurement protocol

- **Units and order.** Each (backend, GPU config) is a *unit* with a freshly started server. Units, and tiers within a unit, run in seeded-random order (`--seed`, recorded) to avoid order and thermal-drift bias.
- **Warm-up, then repetitions.** Each tier gets one untimed warm-up request, then `--repeats` (default 5) timed requests. Every repetition is written as its own row to `results/runs/`, and cells are reported as mean ± 95% CI (Student t) with CV.
- **No prompt caching.** Every request carries a random nonce prefix and llama.cpp gets `cache_prompt: false`, so prefill/TTFT are never served from KV cache. `cached_n` is recorded to prove it.
- **Fixed output length.** llama.cpp requests use `ignore_eos`, so every request generates exactly `max_tokens` and decode speed isn't confounded by output length. Ollama has no equivalent; its `n_generated` is recorded and `hit_max_frac` reported.
- **Thermal control.** Before each unit the harness pauses (`--cooldown`) and waits until all GPUs are ≤ `--max-start-temp` °C. Temperatures, SM clocks and NVML throttle reasons are recorded per repetition.
- **Decode tok/s** comes from the backend's internal timer (`eval_duration` in Ollama, `predicted_per_second` in llama.cpp) — it excludes prompt processing.
- **Telemetry** (every 100 ms per repetition): VRAM, power, temperature, SM clock and PCIe link gen/width per GPU, plus PCIe RX/TX throughput and process RSS. `observed_gpus` lists GPUs whose VRAM rose > 512 MiB above the pre-load baseline — where the model actually ran. Models are flagged as **CPU-spilling** when peak VRAM > 14.5 GB on a single-GPU config.
- **Energy** is the NVML hardware energy-counter delta per request, summed over all GPUs (the idle second card is included — it's part of the rig's cost). Decode J/token uses mean power over the decode window.
- **Bandwidth** uses bytes actually read per token (from the GGUF header), so MoE models no longer exceed the hardware peak. Tensor split is compared against 2× single-GPU peak; layer split against 1×.
- **RAM** is the inference process's peak RSS, not whole-system memory. For llama.cpp this is the exact PID `run.py` spawns; for Ollama it's the heaviest process named `ollama`. Not comparable across backends: llama.cpp's mmap'd weights count toward RSS, Ollama's don't.
- **Failures are data.** Server crashes and request errors are recorded as rows (`ok=False`, `error`, `error_phase`) and reported at the end of the session, not dropped.

### Reproducibility

- Every session writes `results/runs/<session>.env.json`: driver, CUDA, GPU UUIDs/VBIOS/power limits/PCIe capabilities, llama.cpp and Ollama versions, harness git commit (flagged if dirty), and the full CLI arguments.
- `temperature=0` and `seed=0` for all requests.
- All runs on the same machine with no other GPU workloads active.

> Dashboard data collected before this protocol (2026-07 sessions) used 2 repetitions, warm-cache run-2 prefill, variable output length and GGUF-size bandwidth. Treat it as preliminary.

---

## How it works

1. **You provide** a model ID (from the registry in `src/run.py`), a GGUF file on disk, and which backend/GPU configuration(s) to test.
2. **The CLI drives** the benchmark: it starts Ollama or a `llama-server` process with the requested GPU split, sends each prompt tier (chat/RAG/longdoc/code) twice, and samples VRAM/GPU power (via `pynvml`) and process RAM (via `psutil`) every 250 ms while the request runs.
3. **Raw output** goes to a timestamped log; parsed metrics (decode tok/s, prefill tok/s, TTFT, bandwidth, power, RAM) go to a per-session CSV and get appended to the master `results/results.csv`.
4. **The dashboard generator** reads all metrics CSVs, averages the latest 2 runs per (model, config, tier) cell, and writes a self-contained `docs/index.html` you can publish via GitHub Pages.

---

## Quick start

### Prerequisites

- [llama.cpp](https://github.com/ggml-org/llama.cpp) built with CUDA
- GGUF files downloaded to one directory
- Optional: [Ollama](https://ollama.com) running (`ollama serve`) — skipped automatically if absent

### Install and configure

```bash
pip install -r requirements.txt
cp config.example.toml config.toml     # set GGUF_DIR and LLAMACPP_BIN (or env vars of the same name)
python src/run.py models               # checks every GGUF is found and reads its architecture
```

### Adding a model

Add a `[[model]]` block (id, name, file) to [`models.toml`](models.toml) — that's the only change needed. MoE-ness, layer count, hidden size and bytes read per token are read from the GGUF header. The dashboard picks the model up automatically; row order follows `models.toml`.

### Register models with Ollama

```bash
python src/run.py register
```

### Run all benchmarks

```bash
python src/run.py run-all
```

Produces, per session:
- `results/runs/<session>.csv` — one row per repetition (and per failure), full telemetry
- `results/runs/<session>.env.json` — hardware/software manifest
- `results/metrics/<session>.csv` — per-cell mean + 95% CI (dashboard input)
- `results/logs/<session>.log` — full human-readable output
- `results/results.csv` — legacy master CSV (appended)

---

## Usage

### Benchmark a single model

```bash
python src/run.py bench qwen3.5-9b-q4
python src/run.py bench qwen3.5-9b-q4 --backend llamacpp --gpu-configs dual dual_tensor
python src/run.py bench qwen3.5-9b-q4 --tiers chat code
python src/run.py bench qwen3.5-9b-q4 --backend llamacpp --flash-attn off --mmap off
python src/run.py bench qwen3.5-9b-q4 --repeats 10 --seed 42
python src/run.py run-all --models qwen3.5-9b-q4 gemma4-12b-qat --backend llamacpp
```

### Prompt-length sweep

```bash
python src/run.py sweep qwen3.5-9b-q4                                   # 128 … 8192-token prompts
python src/run.py sweep qwen3.5-9b-q4 --lengths 512 4096 --gpu-configs single0 dual_tensor
```

llama.cpp only. Sends exact-length token prompts (raw `/completion`, no chat template) and generates `--gen-tokens` (128) per request, so prefill speed and TTFT can be compared across GPU configs as prompt length grows. Context is sized once per server for the longest prompt, so VRAM is constant across lengths. Results land in the same `results/runs/` files with tier `pp<N>`; summarise with `python src/summarize.py`. Defaults to `single0 dual dual_tensor`.

Protocol flags (`bench`, `run-all` and `sweep`): `--repeats` (5), `--warmup` (1), `--seed`, `--no-shuffle`, `--cooldown` (10 s), `--max-start-temp` (55 °C), `--ignore-eos {on,off}` (on).

`--flash-attn {on,off}` and `--mmap {on,off}` (both default `on`) toggle llama.cpp's flash attention and mmap for weight loading — useful for isolating their effect on decode speed, TTFT, and RAM. Recorded per-run but not yet a dashboard column; compare via `python src/run.py results`.

### Generate the dashboard

```bash
python src/generate_report.py
```

Writes `docs/index.html`. Reads all CSVs in `results/metrics/`, averages the latest 2 runs per cell.

### Other commands

```bash
python src/run.py gpus          # show GPU VRAM and PCIe link state
python src/run.py models        # list models, file status and architecture
python src/run.py env           # print the environment manifest
python src/summarize.py         # mean ± 95% CI per cell from results/runs/ → results/summary.csv
python src/summarize.py --rebuild-metrics   # regenerate metrics CSVs (e.g. after a crashed session)
python src/model_info.py        # architecture table from GGUF headers
python src/run.py results       # legacy results table
python src/log2csv.py           # re-parse legacy (pre-v2) logs → metrics CSVs
python src/analyze.py           # legacy analysis report from results.csv
```

Model paths (`GGUF_DIR`) and the llama.cpp binary (`LLAMACPP_BIN`) are set in `config.toml` or environment variables.

---

## Repository layout

```
src/
  run.py                benchmark CLI (main entry point)
  config.py             paths (config.toml / env) + model registry loader
  telemetry.py          per-repetition GPU/PCIe/energy sampler + environment manifest
  model_info.py         architecture facts from GGUF headers (cached)
  summarize.py          per-cell statistics (mean, 95% CI, CV) + dashboard metrics CSVs
  log2csv.py            parse legacy .log files → metrics CSVs
  generate_report.py    build docs/index.html dashboard
  analyze.py            legacy terminal analysis from results.csv

models.toml             model registry — the one place to add a model
config.example.toml     machine paths template (copy to config.toml)
dracula_ch1.txt         prompt source text
requirements.txt

docs/
  index.html            GitHub Pages dashboard (auto-generated)

results/
  runs/                 per-repetition CSVs + environment manifests (committed)
  metrics/              per-session, per-cell CSVs — dashboard input (committed)
  logs/                 human-readable .log files per session
  model_info.json       cached GGUF architecture facts
  results.csv           legacy master CSV (appended)

reports/
  benchmark_results_template.md   report template (fill by hand per run)
  benchmark_results_YYYYMMDD.md   timestamped result snapshots
```

---

## Dashboard (GitHub Pages)

The dashboard at `docs/index.html` is a self-contained static page — no server needed.

To publish:
1. Push this repo to GitHub
2. Go to **Settings → Pages → Source**: branch `main`, folder `/docs`
3. Dashboard will be live at `https://aniruddh-jammoria.github.io/eval-dual-GPU/`

Features: metric switcher (Decode · Prefill · TTFT · Bandwidth · GPU Power · RAM), tier tabs (chat · RAG · longdoc · code), heat-mapped cells, MoE and CPU-spill annotations, run-count badges.

---

## Changelog & development notes

- [`CHANGELOG.md`](CHANGELOG.md) — notable changes, most recent first.
- [`docs/DEVLOG.md`](docs/DEVLOG.md) — development history and reasoning; significant decisions are recorded as ADRs in [`docs/adr/`](docs/adr/).
