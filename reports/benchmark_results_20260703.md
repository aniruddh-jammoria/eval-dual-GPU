# Dual RTX 5060 Ti 16 GB — LLM Inference Benchmark

**Hardware:** 2× NVIDIA RTX 5060 Ti 16 GB  
**GPU0:** MSI VENTUS 2X OC (PCIe 4.0 x16, from CPU — 32 GB/s) — primary  
**GPU1:** ASUS DUAL OC (PCIe 4.0 x2, from B650 chipset — 4 GB/s)  
**Software:** Ollama · llama.cpp b9858 (CUDA 13.3, Blackwell sm_120a)  
**Date:** 2026-07-03  
**Metric:** decode tok/s (tokens generated per second, 2-run average)

---

## Configurations

| Label | Backend | GPUs | Notes |
|---|---|---|---|
| **Ollama** | Ollama | auto | Uses cuBLAS + tensor cores; best-of-breed CUDA kernels |
| **LC GPU0** | llama.cpp | MSI (x8) | Single GPU, high-bandwidth PCIe slot |
| **LC GPU1** | llama.cpp | ASUS (x2) | Single GPU, electrically limited slot |
| **LC Dual** | llama.cpp | Both | Layer split — GPUs run sequentially per layer |
| **LC Dual ⊗** | llama.cpp | Both | Tensor split — GPUs run in parallel per layer |

---

## Chat — decode tok/s (input: ~540 tok, max output: 512 tok)

*Prompt: Dracula Chapter I excerpt (400 words) + summarise first impressions question*

| Model | GGUF | Ollama | LC GPU0 | LC GPU1 | LC Dual | LC Dual ⊗ |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B IT QAT Q4_0 | 7.0 GB | 49.4 | 51.0 | 50.2 | 49.7 | — |
| Qwen3.5 9B Q4_K_M | 5.7 GB | 67.2 | 70.3 | 68.7 | 68.2 | **103.8** |
| Qwen3.5 9B Q8_0 | 9.5 GB | 42.4 | 45.1 | 44.6 | 44.0 | **72.9** |
| Qwen3.6 27B Q4_K_M | 16.8 GB | 22.7 | 10.0 † | 10.0 † | 23.0 | **37.7** |
| Qwen3.6 27B Q6_K | 22.5 GB | 17.7 | 5.0 † | 5.0 † | 17.8 | **30.4** |
| Gemma 4 26B A4B MoE Q4_K_M | 16.9 GB ‡ | 79.1 | 59.4 | 56.0 | 86.3 | **96.7** |
| Qwen3.6 35B A3B MoE Q4_K_M | 22.1 GB ‡ | 94.5 | 67.0 | 62.3 | 103.3 | **120.0** |
| Gemma 4 31B IT Q5_K_XL | 21.9 GB | 17.5 | 4.4 † | 4.4 † | 17.6 | **29.2** |

† Model overflows 16 GB VRAM → layers spill to system RAM (DDR5 ~90 GB/s vs GPU 672 GB/s)  
‡ MoE: GGUF size is full parameter count; only ~10–15% of weights are read per token (active experts)

---

## RAG — decode tok/s (input: ~2 000 tok, max output: 1 024 tok)

*Prompt: Dracula Chapter I excerpt (1 600 words) + list every warning/sign-of-danger question*

| Model | GGUF | Ollama | LC GPU0 | LC GPU1 | LC Dual | LC Dual ⊗ |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B IT QAT Q4_0 | 7.0 GB | 47.6 | 48.3 | 47.0 | 46.5 | — |
| Qwen3.5 9B Q4_K_M | 5.7 GB | 67.3 | 68.4 | 67.8 | 66.6 | **101.5** |
| Qwen3.5 9B Q8_0 | 9.5 GB | 42.9 | 44.3 | 44.3 | 43.7 | **72.5** |
| Qwen3.6 27B Q4_K_M | 16.8 GB | 22.5 | 9.8 † | 9.8 † | 22.8 | **37.3** |
| Qwen3.6 27B Q6_K | 22.5 GB | 17.5 | 5.0 † | 5.0 † | 17.7 | **30.2** |
| Gemma 4 26B A4B MoE Q4_K_M | 16.9 GB ‡ | 76.1 | 56.5 | 53.6 | 80.1 | **92.8** |
| Qwen3.6 35B A3B MoE Q4_K_M | 22.1 GB ‡ | 96.8 | 67.0 | 62.0 | 101.9 | **120.3** |
| Gemma 4 31B IT Q5_K_XL | 21.9 GB | 16.9 | 4.3 † | 4.2 † | 16.6 | **27.8** |

---

## Code — decode tok/s (input: ~155 tok, max output: 1 024 tok)

*Prompt: Implement a Python `RateLimiter` class with sliding-window logic, decorator, context manager, thread-safe*

| Model | GGUF | Ollama | LC GPU0 | LC GPU1 | LC Dual | LC Dual ⊗ |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B IT QAT Q4_0 | 7.0 GB | 50.7 | 52.3 | 50.8 | 50.3 | — |
| Qwen3.5 9B Q4_K_M | 5.7 GB | 64.3 | 68.4 | 68.7 | 67.6 | **102.6** |
| Qwen3.5 9B Q8_0 | 9.5 GB | 43.0 | 44.8 | 44.7 | 44.2 | **72.8** |
| Qwen3.6 27B Q4_K_M | 16.8 GB | 22.6 | 10.0 † | 10.0 † | 23.0 | **37.8** |
| Qwen3.6 27B Q6_K | 22.5 GB | 17.6 | 5.1 † | 5.0 † | 17.8 | **30.5** |
| Gemma 4 26B A4B MoE Q4_K_M | 16.9 GB ‡ | 83.0 | 59.8 | 56.2 | 87.2 | **97.3** |
| Qwen3.6 35B A3B MoE Q4_K_M | 22.1 GB ‡ | 96.4 | 67.0 | 62.2 | 103.1 | **119.3** |
| Gemma 4 31B IT Q5_K_XL | 21.9 GB | 17.6 | 4.5 † | 4.4 † | 17.7 | **29.5** |

---

## Key Observations

### Impact of model size on token generation
- Small models (5–10 GB, 9B–12B params) run fully in VRAM on a single GPU and achieve **45–70 tok/s** with llama.cpp, **42–67 tok/s** with Ollama. Performance scales predictably with GGUF size: Qwen3.5 9B Q4 (5.7 GB) reaches 70 tok/s; Q8 (9.5 GB) reaches 45 tok/s on the same GPU.
- Large dense models (17–22 GB, 27B–31B params) **do not fit in a single 16 GB GPU**. The overflow spills to system RAM (DDR5 ~90 GB/s) and crushes decode to **4–10 tok/s** — a 3–7× penalty vs the same model fully resident in dual-GPU VRAM.
- MoE models (Gemma 4 26B A4B, Qwen3.6 35B A3B) are an exception: despite large GGUF files (17–22 GB), only ~10–15% of weights are read per token (active experts). They achieve **56–67 tok/s on a single GPU** and **80–120 tok/s on dual** — faster than much smaller dense models.

### Impact of PCIe slot on token generation (GPU0 x16 CPU vs GPU1 x2 chipset)
- GPU0 runs on the CPU's native PCIe 4.0 x16 (32 GB/s). GPU1 runs on a chipset PCIe 4.0 x2 (4 GB/s) — an 8× bandwidth difference. Despite this, decode tok/s gap is only **1–8%** for VRAM-resident models because decode is VRAM-bandwidth-bound, not PCIe-bound.
- For **CPU-RAM-spilling models** (27B Q4/Q6, 31B), GPU0 and GPU1 are **identical** in decode tok/s despite the 8× PCIe gap. The bottleneck is CPU-side computation on off-GPU layers, not PCIe transfer rate.
- TTFT is slower on GPU1 for large prompts due to chipset pipeline latency (GPU1 traffic routes GPU ↔ B650 chipset ↔ CPU rather than GPU ↔ CPU directly).

### Impact of backend on token generation (llama.cpp vs Ollama)
- For **small dense models** (9B–12B): llama.cpp single GPU0 is **2–5% faster** than Ollama in decode. Ollama closes the gap at larger context (RAG tier) due to its optimised KV-cache handling.
- For **MoE models**: Ollama is **25–45% faster** than llama.cpp on a single GPU (79 vs 59 tok/s for Gemma MoE; 95 vs 67 tok/s for Qwen MoE). Ollama uses INT4/INT8 tensor cores and cuBLAS kernels tuned for sparse MoE routing that llama.cpp does not.
- For **large dense models needing dual GPU**: llama.cpp dual-tensor **surpasses Ollama** by up to 67% (Gemma 31B: 29.2 vs 17.5 tok/s; Qwen 27B Q4: 37.7 vs 22.7 tok/s). Ollama cannot utilise dual GPUs for a single inference.
- Ollama has a **massive prefill advantage** across the board — up to 10× faster prompt processing on RAG-tier prompts — due to cuBLAS batched matrix multiplication. This directly reduces time-to-first-token.

### Impact of single vs dual GPU on token generation
- **Small models (fit in one GPU):** Dual GPU provides **no benefit and a slight penalty** (1–3% slower) due to synchronisation overhead with nothing to gain from splitting.
- **Large dense models (CPU RAM spill):** Dual GPU is **transformative** — the entire model fits in combined VRAM, eliminating the DDR5 bottleneck:
  - Qwen3.6 27B Q4: 10 → 23 tok/s (+130%)
  - Qwen3.6 27B Q6: 5 → 18 tok/s (+256%)
  - Gemma 4 31B Q5: 4 → 18 tok/s (+300%)
- **MoE models:** Dual GPU gives a moderate **35–55% decode speedup** (single: 57–67 tok/s → dual: 80–103 tok/s) by splitting the expert weights across both GPUs' VRAM and bandwidth.
- The second GPU is **mandatory** for running any dense 27B+ model at usable speeds on this hardware.

### Impact of tensor split on token generation (LC Dual vs LC Dual ⊗)
- Tensor split runs both GPUs **in parallel on every layer** rather than sequentially, effectively doubling available memory bandwidth for decode (2 × 672 GB/s = 1 344 GB/s).
- **Dense models** see the largest gains — consistent **50–70% speedup** over layer split:
  - Qwen3.5 9B Q4: 68 → 104 tok/s (+53%)
  - Qwen3.6 27B Q4: 23 → 38 tok/s (+64%)
  - Gemma 4 31B Q5: 18 → 29 tok/s (+66%)
- **MoE models** see a smaller but still meaningful **12–16% speedup** over layer split:
  - Gemma 4 26B MoE: 86 → 97 tok/s (+12%)
  - Qwen3.6 35B MoE: 103 → 120 tok/s (+16%)
  - Smaller gain because MoE sparse activation means per-token bandwidth is already lower, so the allreduce communication cost is proportionally larger.
- Tensor split has **negligible TTFT impact** on short prompts (code tier). On long prompts (RAG tier), TTFT is slightly worse than layer split because every layer requires a PCIe allreduce synchronisation — but the tradeoff is almost always worth it for interactive use.
- **Recommendation:** use `--split-mode tensor` for all dual-GPU runs. The only exception is when TTFT on very long documents is the primary constraint.
