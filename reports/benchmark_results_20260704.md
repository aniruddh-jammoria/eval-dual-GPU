# Dual RTX 5060 Ti 16 GB — LLM Inference Benchmark

Date run: 04-Jul-2026

---

### Hardware specifications
**GPU0:** MSI VENTUS 2X OC (PCIe 4.0 x16, from CPU — 32 GB/s) — primary  
**GPU1:** ASUS DUAL OC (PCIe 4.0 x2, from B650 chipset — 4 GB/s)  
**Motherboard:** MSI MAG B650 TOMAHAWK WIFI  
**CPU:** AMD Ryzen 9 7900  
**Memory:** 32 GB DDR5  
**Hardware:** 2× NVIDIA RTX 5060 Ti 16 GB (Blackwell sm_120a)  

### Software specifications
**Ollama:** Ollama (latest)  
**llama.cpp:** llama.cpp b9858 (CUDA 13.3)  

---

### Inference configurations

| Label | Backend | GPU config | Notes |
|---|---|---|---|
| **OL** | Ollama | GPU0 (auto) | Ollama picks the device; runs on GPU0. cuBLAS + tensor-core kernels. |
| **LC-S0** | llama.cpp | Single GPU0 | MSI card, PCIe 4.0 x16 from CPU. |
| **LC-S1** | llama.cpp | Single GPU1 | ASUS card, PCIe 4.0 x2 from chipset. |
| **LC-Dual** | llama.cpp | Dual | Layer split — GPUs run sequentially, one holds each layer range. |
| **LC-Dual⊗** | llama.cpp | Dual | Tensor split (`--split-mode tensor`) — both GPUs run every layer in parallel. |

> *dual_21 (2:1 layer split) is omitted — it was not part of the main run-all.*

---

### Response configurations

| Tier | Input tok | Output tok | Relevance | Task |
|---|---:|---:|---|---|
| `chat` | ~540 | 512 | Short conversational context — a typical single chat turn. | Summarise Harker's first impressions from a 400-word Dracula excerpt. |
| `rag` | ~2 000 | 1 024 | Retrieval-augmented: large input context, focused extraction. | List every warning/sign of danger in a 1 600-word Dracula excerpt. |
| `code` | ~155 | 1 024 | Code generation: short prompt, long structured output. | Implement a thread-safe sliding-window `RateLimiter` class. |

---

### Metrics

| Metric | Definition | Formula | Calculated from |
|---|---|---|---|
| Decode tok/s | Tokens generated per second during the generation phase (memory-bandwidth bound). | `n_generated / decode_time` | Backend internal timer — Ollama `eval_duration`, llama.cpp `predicted_per_second`. |
| Prefill tok/s | Input (prompt) tokens processed per second before generation (compute bound). | `n_prompt / prefill_time` | Ollama `prompt_eval_duration`, llama.cpp `prompt_per_second`. |
| TTFT (s) | Time to first token — latency of the prefill phase. | `prefill_time` | Ollama `prompt_eval_duration`, llama.cpp `prompt_ms`. |
| Bandwidth (GB/s) | Effective memory bandwidth exercised during decode. | `gguf_size_GB × decode_tok_s` | Derived. For MoE this exceeds rated peak — only active experts are read per token. |
| Peak power (W) | Peak combined GPU board power during inference (supported GPUs only). | `max Σ nvmlDeviceGetPowerUsage` | NVML sampled at 2 Hz. GPU0 (MSI) returns NotSupported — see energy-table caveat. |

---

### Prefill and Decode token generation

#### Tier `chat` (input: ~540 tok, max output: 512 tok)

**Decode tok/s** (higher = faster generation)

| Model | Size (GB) | OL | LC-S0 | LC-S1 | LC-Dual | LC-Dual⊗ |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma** | | | | | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | 49.4 | **51.0** | 50.2 | 49.7 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) ‡ | 16.9 | 79.1 | 59.4 † | 56.0 | 86.3 | **96.7** |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | 17.5 | 4.4 † | 4.4 | 17.6 | **29.2** |
| **Qwen** | | | | | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | 67.0 | 70.1 | 68.6 | 68.0 | **103.1** |
| Qwen3.5 9B (Q8_0) | 9.5 | 42.4 | 45.1 | 44.6 | 44.0 | **72.9** |
| Qwen3.6 27B (Q4_K_M) | 16.8 | 22.7 | 10.0 † | 10.0 | 23.0 | **37.7** |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) ‡ | 22.1 | 94.5 | 67.0 † | 62.3 | 103.3 | **120.0** |
| Qwen3.6 27B (Q6_K) | 22.7 | 17.7 | 5.0 † | 5.0 | 17.8 | **30.4** |

† single-GPU config spills over 16 GB VRAM into system RAM  
‡ Mixture-of-Experts — only active experts read per token

**Prefill tok/s** (higher = faster prompt processing)

| Model | Size (GB) | OL | LC-S0 | LC-S1 | LC-Dual | LC-Dual⊗ |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma** | | | | | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | **9123** | 1882 | 1854 | 1939 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) ‡ | 16.9 | **7644** | 807 † | 395 | 2287 | 1391 |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | **730** | 274 † | 105 | 660 | 531 |
| **Qwen** | | | | | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | **16822** | 1294 | 1158 | 1196 | 803 |
| Qwen3.5 9B (Q8_0) | 9.5 | **8641** | 1330 | 1192 | 1217 | 803 |
| Qwen3.6 27B (Q4_K_M) | 16.8 | **794** | 238 † | 126 | 392 | 308 |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) ‡ | 22.1 | **11222** | 205 † | 146 | 930 | 754 |
| Qwen3.6 27B (Q6_K) | 22.7 | **682** | 129 † | 69 | 344 | 293 |

† single-GPU config spills over 16 GB VRAM into system RAM  
‡ Mixture-of-Experts — only active experts read per token

_Note: Ollama prefill is inflated by warm KV-cache reuse on the 2nd timed run — treat OL prefill as indicative, not directly comparable._

**TTFT (s)** (lower = less latency to first token)

| Model | Size (GB) | OL | LC-S0 | LC-S1 | LC-Dual | LC-Dual⊗ |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma** | | | | | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | **0.060** | 0.280 | 0.290 | 0.270 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) ‡ | 16.9 | **0.070** | 0.760 † | 1.350 | 0.240 | 0.390 |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | **0.730** | 1.950 † | 5.050 | 0.810 | 1.000 |
| **Qwen** | | | | | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | **0.030** | 0.120 | 0.140 | 0.140 | 0.200 |
| Qwen3.5 9B (Q8_0) | 9.5 | **0.060** | 0.120 | 0.140 | 0.140 | 0.200 |
| Qwen3.6 27B (Q4_K_M) | 16.8 | 0.680 | 0.710 † | 1.270 | **0.410** | 0.520 |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) ‡ | 22.1 | **0.050** | 0.850 † | 1.190 | 0.170 | 0.220 |
| Qwen3.6 27B (Q6_K) | 22.7 | 0.790 | 1.290 † | 2.320 | **0.470** | 0.540 |

† single-GPU config spills over 16 GB VRAM into system RAM  
‡ Mixture-of-Experts — only active experts read per token


#### Tier `rag` (input: ~2 000 tok, max output: 1 024 tok)

**Decode tok/s** (higher = faster generation)

| Model | Size (GB) | OL | LC-S0 | LC-S1 | LC-Dual | LC-Dual⊗ |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma** | | | | | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | 47.6 | **48.3** | 47.0 | 46.5 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) ‡ | 16.9 | 76.1 | 56.5 † | 53.6 | 80.1 | **92.8** |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | 16.9 | 4.3 † | 4.2 | 16.6 | **27.8** |
| **Qwen** | | | | | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | 67.3 | 68.4 | 67.8 | 66.6 | **101.5** |
| Qwen3.5 9B (Q8_0) | 9.5 | 42.9 | 44.3 | 44.3 | 43.7 | **72.5** |
| Qwen3.6 27B (Q4_K_M) | 16.8 | 22.5 | 9.8 † | 9.8 | 22.8 | **37.3** |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) ‡ | 22.1 | 96.8 | 67.0 † | 62.0 | 101.9 | **120.3** |
| Qwen3.6 27B (Q6_K) | 22.7 | 17.5 | 5.0 † | 5.0 | 17.7 | **30.2** |

† single-GPU config spills over 16 GB VRAM into system RAM  
‡ Mixture-of-Experts — only active experts read per token

**Prefill tok/s** (higher = faster prompt processing)

| Model | Size (GB) | OL | LC-S0 | LC-S1 | LC-Dual | LC-Dual⊗ |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma** | | | | | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | **44983** | 1106 | 1084 | 1100 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) ‡ | 16.9 | **67714** | 714 † | 551 | 1663 | 901 |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | **2157** | 246 † | 131 | 388 | 318 |
| **Qwen** | | | | | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | **5346** | 1452 | 1394 | 1409 | 924 |
| Qwen3.5 9B (Q8_0) | 9.5 | **4685** | 1575 | 1505 | 1529 | 963 |
| Qwen3.6 27B (Q4_K_M) | 16.8 | **2897** | 339 † | 238 | 454 | 351 |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) ‡ | 22.1 | **4644** | 474 † | 365 | 1503 | 1114 |
| Qwen3.6 27B (Q6_K) | 22.7 | **2616** | 240 † | 154 | 397 | 334 |

† single-GPU config spills over 16 GB VRAM into system RAM  
‡ Mixture-of-Experts — only active experts read per token

_Note: Ollama prefill is inflated by warm KV-cache reuse on the 2nd timed run — treat OL prefill as indicative, not directly comparable._

**TTFT (s)** (lower = less latency to first token)

| Model | Size (GB) | OL | LC-S0 | LC-S1 | LC-Dual | LC-Dual⊗ |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma** | | | | | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | **0.040** | 0.470 | 0.480 | 0.480 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) ‡ | 16.9 | **0.030** | 0.730 † | 0.950 | 0.310 | 0.580 |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | **0.930** | 2.170 † | 4.000 | 1.350 | 1.640 |
| **Qwen** | | | | | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | **0.380** | **0.380** | 0.400 | 0.390 | 0.600 |
| Qwen3.5 9B (Q8_0) | 9.5 | 0.430 | **0.350** | 0.370 | 0.360 | 0.580 |
| Qwen3.6 27B (Q4_K_M) | 16.8 | **0.700** | 1.660 † | 2.350 | 1.220 | 1.580 |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) ‡ | 22.1 | 0.440 | 1.200 † | 1.550 | **0.370** | 0.500 |
| Qwen3.6 27B (Q6_K) | 22.7 | **0.780** | 2.370 † | 3.620 | 1.400 | 1.660 |

† single-GPU config spills over 16 GB VRAM into system RAM  
‡ Mixture-of-Experts — only active experts read per token


#### Tier `code` (input: ~155 tok, max output: 1 024 tok)

**Decode tok/s** (higher = faster generation)

| Model | Size (GB) | OL | LC-S0 | LC-S1 | LC-Dual | LC-Dual⊗ |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma** | | | | | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | 50.7 | **52.3** | 50.8 | 50.3 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) ‡ | 16.9 | 83.0 | 59.8 † | 56.2 | 87.2 | **97.3** |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | 17.6 | 4.5 † | 4.4 | 17.7 | **29.5** |
| **Qwen** | | | | | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | 64.3 | 68.4 | 68.7 | 67.6 | **102.6** |
| Qwen3.5 9B (Q8_0) | 9.5 | 43.0 | 44.8 | 44.7 | 44.2 | **72.8** |
| Qwen3.6 27B (Q4_K_M) | 16.8 | 22.6 | 10.0 † | 10.0 | 23.0 | **37.8** |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) ‡ | 22.1 | 96.4 | 67.0 † | 62.2 | 103.1 | **119.3** |
| Qwen3.6 27B (Q6_K) | 22.7 | 17.6 | 5.1 † | 5.0 | 17.8 | **30.5** |

† single-GPU config spills over 16 GB VRAM into system RAM  
‡ Mixture-of-Experts — only active experts read per token

**Prefill tok/s** (higher = faster prompt processing)

| Model | Size (GB) | OL | LC-S0 | LC-S1 | LC-Dual | LC-Dual⊗ |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma** | | | | | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | **2356** | 1229 | 1140 | 1263 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) ‡ | 16.9 | **2871** | 380 † | 160 | 1152 | 791 |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | **821** | 96 † | 36 | 436 | 339 |
| **Qwen** | | | | | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | **2654** | 797 | 670 | 701 | 488 |
| Qwen3.5 9B (Q8_0) | 9.5 | **2446** | 782 | 683 | 722 | 462 |
| Qwen3.6 27B (Q4_K_M) | 16.8 | **1150** | 109 † | 50 | 264 | 198 |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) ‡ | 22.1 | **3188** | 109 † | 73 | 460 | 370 |
| Qwen3.6 27B (Q6_K) | 22.7 | **978** | 47 † | 26 | 238 | 203 |

† single-GPU config spills over 16 GB VRAM into system RAM  
‡ Mixture-of-Experts — only active experts read per token

_Note: Ollama prefill is inflated by warm KV-cache reuse on the 2nd timed run — treat OL prefill as indicative, not directly comparable._

**TTFT (s)** (lower = less latency to first token)

| Model | Size (GB) | OL | LC-S0 | LC-S1 | LC-Dual | LC-Dual⊗ |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma** | | | | | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | **0.070** | 0.130 | 0.140 | 0.120 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) ‡ | 16.9 | **0.060** | 0.510 † | 0.990 | 0.150 | 0.210 |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | **0.200** | 1.780 † | 4.390 | 0.370 | 0.470 |
| **Qwen** | | | | | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | **0.070** | **0.070** | 0.080 | 0.080 | 0.110 |
| Qwen3.5 9B (Q8_0) | 9.5 | **0.060** | 0.070 | 0.080 | 0.080 | 0.110 |
| Qwen3.6 27B (Q4_K_M) | 16.8 | **0.130** | 0.500 † | 1.030 | 0.200 | 0.260 |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) ‡ | 22.1 | **0.050** | 0.630 † | 0.940 | 0.110 | 0.140 |
| Qwen3.6 27B (Q6_K) | 22.7 | **0.160** | 1.090 † | 2.020 | 0.230 | 0.260 |

† single-GPU config spills over 16 GB VRAM into system RAM  
‡ Mixture-of-Experts — only active experts read per token

---

### Peak energy consumption

> **Power-measurement caveat.** GPU0 (MSI VENTUS RTX 5060 Ti) does not expose power telemetry to NVML (`nvmlDeviceGetPowerUsage` → *NotSupported*; `nvidia-smi` also reports `N/A`). GPU1 (ASUS DUAL) reports fine. Therefore only **LC-S1** — where GPU1 is the active card — yields a trustworthy figure. `OL` and `LC-S0` would show only idle GPU1 draw (~4 W), and dual configs capture GPU1's share only, so they are omitted here rather than reported misleadingly.

Peak GPU board power on **LC-S1** (GPU1 active), range across tiers.

| Model | Size (GB) | LC-S1 peak power (W) |
|---|---:|---:|
| **Gemma** | | |
| Gemma 4 12B IT QAT (Q4_0) | 7.0 | — |
| Gemma 4 26B A4B IT MoE UD (Q4_K_M) | 16.9 | — |
| Gemma 4 31B IT UD (Q5_K_XL) | 22.0 | — |
| **Qwen** | | |
| Qwen3.5 9B (Q4_K_M) | 5.7 | 153 |
| Qwen3.5 9B (Q8_0) | 9.5 | — |
| Qwen3.6 27B (Q4_K_M) | 16.8 | — |
| Qwen3.6 35B A3B MoE UD (Q4_K_M) | 22.1 | — |
| Qwen3.6 27B (Q6_K) | 22.7 | — |

---

### Key Observations

**Model size — dense models above ~16 GB spill to system RAM on one GPU.**
- On a single GPU, oversized dense models collapse to 4.4–10.0 tok/s. Adding the second GPU restores them: Gemma 4 31B IT UD goes 4.4 → 29.2 tok/s (**6.6×**).
- MoE models are the exception: 59–67 tok/s on a single GPU despite large files, because only active experts are read per token.

**PCIe slot — decode is bandwidth-bound, so the x2 chipset link barely matters.**
- For VRAM-resident models, GPU0 (x16) leads GPU1 (x2) by only 1–2% in decode.
- Prefill is far more slot-sensitive: GPU0 processes prompts 1.0–1.1× faster than GPU1.
- For spilling models, GPU0 and GPU1 are identical — the bottleneck moves off the PCIe link.

**Backend — Ollama wins on MoE, llama.cpp dual wins on big dense models.**
- Small dense models: llama.cpp on GPU0 is 3–6% faster than Ollama.
- MoE: Ollama beats single-GPU llama.cpp by 33–41%.
- Large dense: llama.cpp dual-tensor beats Ollama by up to 72% — Ollama can't split one inference across GPUs.

**Single vs dual GPU — mandatory for big dense, marginal for small.**
- Small models: dual layer-split shifts decode by -3% to -2% (no real gain).
- Big dense models: the second GPU is the difference between unusable and usable (see model-size lifts).

**Tensor split — parallel per-layer reads roughly double effective bandwidth.**
- Dense: LC-Dual⊗ is 52–71% faster than layer split.
- MoE: a smaller 12–16% gain — sparse activation limits per-token reads.
- **Recommendation:** default to `--split-mode tensor` for any dual-GPU run.

*Observation figures computed from the `chat` tier.*
