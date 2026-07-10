# Dual RTX 5060 Ti 16 GB — LLM Inference Benchmark

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![CLI](https://img.shields.io/badge/type-CLI-informational)

How much do a second GPU, its split mode, and backend choice (Ollama vs llama.cpp)
actually change local LLM inference performance? This repo benchmarks 8 models across
5 GPU configurations and 2 backends on two RTX 5060 Ti 16 GB cards, measuring decode
speed, prefill speed, latency, memory bandwidth, and power draw for each combination.

Dense models over 16 GB spill from a single GPU into system RAM and crawl at 4–10 tok/s.
This project measures how much a cheap, chipset-slot second GPU recovers, whether layer
split or tensor split wins, and how Ollama and llama.cpp trade off against each other
across configurations.

**[→ Live dashboard](https://aniruddh-jammoria.github.io/eval-dual-GPU/)**

---

## Hardware

| | GPU0 | GPU1 |
|---|---|---|
| Card | MSI VENTUS 2X OC | ASUS DUAL OC |
| Slot | PCIe 4.0 x16 (CPU) — 32 GB/s | PCIe 4.0 x2 (B650 chipset) — 4 GB/s |
| VRAM | 16 GB GDDR7 | 16 GB GDDR7 |
| Bandwidth | 672 GB/s | 672 GB/s |

**Platform:** AMD Ryzen 9 7900 · 32 GB DDR5 · MSI MAG B650 TOMAHAWK WIFI · Windows 11
**Software:** Ollama · llama.cpp b9858 (CUDA 13.3, Blackwell sm_120a)

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
| **Ollama** | Ollama | GPU0 (auto) | cuBLAS + tensor cores |
| **LC GPU0** | llama.cpp | GPU0 only | PCIe 4.0 x16, full CPU lanes |
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
| **Bandwidth (GB/s)** | Effective memory bandwidth used: `GGUF_size × decode_tok_s` |
| **GPU Power (W)** | Average combined GPU power draw during inference |

All metrics are averaged over 2 runs per configuration. Decode tok/s uses the model's internal timing, not wall clock.

---

## Key findings

- **Dense models > 16 GB spill to system RAM** on a single GPU, dropping decode from ~50 tok/s to 4–10 tok/s (DDR5 ~90 GB/s vs GPU 672 GB/s). Dual GPU eliminates the spill.
- **Gemma 4 31B: 4.4 → 29.2 tok/s** with dual tensor split — a **6.6× lift** from the second GPU.
- **MoE models are an exception**: Gemma 4 26B A4B and Qwen 35B A3B reach 56–67 tok/s on a single GPU despite large GGUF sizes because only ~10–15% of weights are read per token (active experts).
- **Tensor split (`--split-mode tensor`)** runs both GPUs in parallel per layer, doubling effective bandwidth (2 × 672 = 1 344 GB/s). Gives **50–70% speedup** over layer split for dense models.
- **Ollama beats llama.cpp for MoE** on a single GPU by 25–45% (cuBLAS sparse routing kernels), but **cannot split a single inference across two GPUs**.
- **The PCIe x2 slot is not a bottleneck for decode**: allreduce data in tensor split mode is ~14 KB per layer per token — negligible over 4 GB/s.

Full tables and observations: [`reports/benchmark_results_20260703.md`](reports/benchmark_results_20260703.md)

---

## How it works

1. **You provide** a model ID (from the registry in `src/run.py`), a GGUF file on disk, and which backend/GPU configuration(s) to test.
2. **The CLI drives** the benchmark: it starts Ollama or a `llama-server` process with the requested GPU split, sends each prompt tier (chat/RAG/longdoc/code) twice, and samples VRAM and GPU power every 250 ms via `pynvml` while the request runs.
3. **Raw output** goes to a timestamped log; parsed metrics (decode tok/s, prefill tok/s, TTFT, bandwidth, power) go to a per-session CSV and get appended to the master `results/results.csv`.
4. **The dashboard generator** reads all metrics CSVs, averages the latest 2 runs per (model, config, tier) cell, and writes a self-contained `docs/index.html` you can publish via GitHub Pages.

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

### Measurement

- Each (model, config, tier) combination runs **2 times**; both runs are recorded and averaged.
- **Decode tok/s** comes from the backend's internal timer (`eval_duration` in Ollama, `predicted_per_second` in llama.cpp timings). This is the standard benchmark metric — it excludes prompt processing time.
- **TTFT / Prefill** uses `prompt_eval_duration`. Run 2's prefill is artificially fast (warm KV cache) — treat it as indicative only.
- VRAM usage is sampled every 250 ms via pynvml. Models are flagged as **CPU-spilling** when peak VRAM > 14.5 GB on a single-GPU config.
- Bandwidth is derived: `bw = GGUF_size_GB × decode_tok_s`. For MoE models this exceeds the GPU's rated peak (expected — only active experts are read per token).

### Reproducibility

- llama.cpp server started fresh for each configuration; Ollama model evicted between runs.
- `temperature=0` for all runs (deterministic output).
- All runs on the same machine with no other GPU workloads active.

---

## Quick start

### Prerequisites

- [Ollama](https://ollama.com) running (`ollama serve`)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built with CUDA — set `LLAMACPP_BIN` in `src/run.py`
- GGUF files downloaded to your model directory — set `GGUF_DIR` in `src/run.py`

### Install

```bash
pip install -r requirements.txt
```

### Register models with Ollama

```bash
python src/run.py register
```

### Run all benchmarks

```bash
python src/run.py run-all
```

Produces:
- `results/logs/<timestamp>_run-all.log` — full human-readable output
- `results/metrics/<timestamp>_run-all.csv` — per-run metrics
- `results/results.csv` — master CSV (appended)

---

## Usage

### Benchmark a single model

```bash
python src/run.py bench qwen3.5-9b-q4
python src/run.py bench qwen3.5-9b-q4 --backend llamacpp --gpu-configs dual dual_tensor
python src/run.py bench qwen3.5-9b-q4 --tiers chat code
```

### Generate the dashboard

```bash
python src/generate_report.py
```

Writes `docs/index.html`. Reads all CSVs in `results/metrics/`, averages the latest 2 runs per cell.

### Other commands

```bash
python src/run.py gpus          # show GPU VRAM state
python src/run.py models        # list models and file status
python src/run.py results       # print results table in terminal
python src/log2csv.py           # re-parse all logs → metrics CSVs
python src/analyze.py           # full analysis report from results.csv
```

Model paths (`GGUF_DIR`) and the llama.cpp binary (`LLAMACPP_BIN`) are set at the top of `src/run.py`.

---

## Repository layout

```
src/
  run.py                benchmark CLI (main entry point)
  log2csv.py            parse .log files → metrics CSVs
  generate_report.py    build docs/index.html dashboard
  analyze.py            terminal analysis from results.csv

dracula_ch1.txt         prompt source text
requirements.txt

docs/
  index.html            GitHub Pages dashboard (auto-generated)

results/
  logs/                 raw .log files per session
  metrics/              per-session .csv files (committed)
  results.csv           master CSV (all runs, appended)

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

Features: metric switcher (Decode · Prefill · TTFT · Bandwidth · GPU Power), tier tabs (chat · RAG · longdoc · code), heat-mapped cells, MoE and CPU-spill annotations, run-count badges.
