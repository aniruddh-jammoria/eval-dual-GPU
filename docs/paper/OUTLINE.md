# Paper outline (tentative)

Status: draft outline, 2026-09-27. Nothing here is a claim yet. Numbers marked *prelim* come
from existing `results/metrics/*.csv` (2–6 sessions per cell, pre-methodology-fixes) and
exist only to shape hypotheses.

**Target:** Zenodo technical report (DOI) first, then arXiv — primary `cs.DC`, cross-list `cs.PF`, `cs.LG`.
**Format:** LaTeX, ~8 pages + appendix, single author. Artifact (code + raw data) archived on Zenodo and cited from the paper.

---

## Working title

> *Tensor Parallelism Over a Chipset x2 Link: Multi-GPU LLM Inference on Asymmetric Consumer PCIe Topologies*

Alternates:
- *Two Cheap GPUs or One Expensive One? An Empirical Study of Split Strategies for Local LLM Inference*
- *When Does the Interconnect Matter? Layer vs. Tensor Splitting on Consumer Dual-GPU Systems*

Pick the title after results land — it should state the finding, not the setup.

---

## The core idea (one paragraph — keep this stable, everything serves it)

Consumer motherboards typically give the second GPU a narrow chipset-attached link (here PCIe 4.0 x2,
~4 GB/s vs 32 GB/s for the primary slot). Conventional wisdom says tensor parallelism — which
all-reduces activations at every layer — should be crippled by such a link, and that layer (pipeline)
splitting is the only sensible choice. Preliminary data contradicts this: at batch size 1, tensor split
over the x2 link is **~50% faster in decode than a single GPU even for models that fit on one card**
(*prelim*: Qwen3.5 9B Q4 ≈ 68 → 103 tok/s), while layer split gives no speedup at all. The paper
explains *why* (per-token all-reduce payload is tiny; decode is latency-, not bandwidth-bound on the
link, and splitting halves weight-read bytes per GPU), *where it breaks down* (prefill, long context,
batch > 1, narrower links), and turns this into guidance for building local inference rigs.

---

## Research questions

| # | Question | Why it matters |
|---|---|---|
| RQ1 | How does split strategy (single / layer / tensor) affect decode and prefill throughput across model sizes and architectures (dense vs MoE)? | Main result |
| RQ2 | How sensitive is each strategy to inter-GPU link bandwidth? | The novel part — needs the link-bandwidth sweep (Exp. B) |
| RQ3 | What is the cost of VRAM overflow on one card vs adding a second card? | Practical "buy a second GPU?" question |
| RQ4 | How close does each config get to a roofline bound, and what explains the gap? | Turns measurements into understanding |
| RQ5 (secondary) | Energy per token and cost per tok/s across configs | Practitioner value |

Out of scope (state explicitly): output quality/accuracy, batch serving throughput at scale (unless Exp. E is done), training, non-NVIDIA hardware.

---

## Sections

### Abstract (~200 words) — write last
Problem → gap → what we did (N models × M configs × link-bandwidth sweep) → 2–3 headline numbers → one-line takeaway → artifact availability.

### 1. Introduction (~1 page)
**Contents**
- Motivation: VRAM is the binding constraint for local LLMs; 32 GB via 2× 16 GB consumer cards costs ~25–30% of a single 32 GB flagship.
- The problem: consumer platforms wire the 2nd slot through the chipset at x2/x4; almost all published multi-GPU inference results assume NVLink or full x16 server topologies.
- The counter-intuitive preliminary finding (tensor split wins over a 4 GB/s link at batch 1).
- Contributions (bulleted, each testable):
  1. Controlled measurement of layer vs tensor splitting on an asymmetric consumer topology across 8 models (dense + MoE, 9B–35B).
  2. A link-bandwidth sweep isolating the interconnect's effect (Exp. B).
  3. An analytical model (roofline + communication cost) that predicts when tensor split pays off.
  4. Open dataset, harness, and dashboard.
- **Figure 1** (the "money plot"): decode tok/s vs model size, one line per config, spill region shaded.

**To do:** nothing until results are final; draft last-but-one.

### 2. Background (~0.75 page)
**Contents**
- LLM inference phases: prefill (compute-bound) vs decode (memory-bandwidth-bound at batch 1). Roofline framing.
- Parallelism strategies: layer/pipeline split (sequential GPUs, activations passed once per boundary) vs tensor split (Megatron-style, all-reduce per layer, both GPUs active).
- Consumer PCIe topology: CPU lanes vs chipset lanes, shared chipset uplink (x4 Gen4 on B650), no P2P on GeForce → transfers bounce through host memory.
- Quantized GGUF formats; MoE active-parameter reads.
- **Figure 2:** system topology diagram (CPU, chipset, two GPUs, link widths).

**To do:** verify whether P2P is available on this platform (`nvidia-smi topo -m`, CUDA `p2pBandwidthLatencyTest`) — this changes the communication model.

### 3. Related work (~0.5–0.75 page)
**Contents** — group by theme, position this paper in one sentence per group:
- Parallelism for LLMs: Megatron-LM (tensor), GPipe/PipeDream (pipeline).
- Inference systems & offloading on constrained hardware: FlexGen, PowerInfer, llama.cpp, vLLM, ExLlama.
- Roofline analysis of LLM inference (e.g., Yuan et al. 2024 survey with roofline).
- Benchmarking methodology: MLPerf Inference; LLM-perf leaderboards.
- Community benchmarks of consumer multi-GPU (llama.cpp discussions, blog posts) — cite as grey literature, note lack of controlled topology.

**To do:** literature pass; build `references.bib`; verify every citation exists and says what we claim (no citing from memory).

### 4. Experimental setup (~1 page)
**Contents**
- Hardware table (CPU, board, RAM, both GPUs with slot/link/vendor), driver version, power limits, clocks, cooling/ambient.
- Software: OS build, CUDA, llama.cpp commit hash, Ollama version, Python harness commit.
- Models table: params, active params (MoE), quant, GGUF bytes, layers, hidden dim (needed for comm model).
- Model selection rationale (target ~12–15 models, chosen to de-confound, not to maximize count):
  - size ladder within one family at a fixed quant (Q4_K_M) — clean size axis for Fig. 1
  - dense 2–3 models with GGUF 13–18 GB — the fits-vs-spills boundary on one card
  - matched dense/MoE pairs at similar total size — tests the MoE tensor-split effect
  - 1–2 additional families (different hidden dim / layer count) — varies comm payload in the analytical model
  - one mid-size model at Q4/Q6/Q8 — isolates bytes-per-token from architecture
- Reference systems (cloud, see Exp. I/J): single 32 GB GPU (RTX 5090) and a symmetric dual-GPU box (both x16). Framed as reference points, compared on tok/s, tok/s per $, tok/s per W — not as controlled like-for-like.
- Configurations table: single0, single1, layer split, tensor split, Ollama (with *actual* observed placement).
- Workloads: controlled microbenchmarks (`llama-bench`: pp512/pp2048, tg128 at depths 0/2k/4k) **plus** application-level tiers (chat/rag/longdoc/code via server). Justify both.
- Metrics with exact definitions and sources (decode tok/s, prefill tok/s, TTFT, energy/token, peak VRAM, host RSS, PCIe throughput).
- Measurement protocol: warm-up, ≥5 repetitions, randomized config order, cache disabled, fixed output length, thermal steady-state check, idle-system checks.
- **Table 1–3:** hardware/software, models, configs.

**To do:** see Methodology work list (M1–M10) below.

### 5. Analytical model (~0.75 page)
**Contents**
- Decode roofline: `tok/s ≤ BW_eff / bytes_read_per_token` (dense: full weights; MoE: active experts + shared).
- Layer split: same bytes, sequential GPUs → no speedup expected (matches *prelim* data: dual ≈ single).
- Tensor split: bytes per GPU halved → up to 2× ideal, minus per-layer all-reduce cost:
  `t_comm ≈ n_layers × k × (latency + payload / link_BW)`, payload = hidden_dim × dtype bytes (per token at batch 1).
- Predict: decode gain is latency-dominated and insensitive to link BW; prefill payload scales with prompt length → link-BW-sensitive.
- Predict crossover batch size / prompt length where the x2 link starts to dominate.

**To do:** derive per-model constants; measure link latency and bandwidth (host-staged copies) to plug in.

### 6. Results (~2.5 pages)
One subsection per RQ, each opening with the takeaway sentence.
- **6.1 Split strategy vs model size (RQ1)** — Fig. 1 + table of decode/prefill per model×config with 95% CIs. Dense vs MoE contrast (*prelim*: MoE gains less from tensor split, ~+17% vs ~+50–65% dense).
- **6.2 Sensitivity to link bandwidth (RQ2)** — Exp. B: throughput vs link BW (x16 Gen4 → Gen1 / x2), per strategy and phase. Predicted: decode flat, prefill slopes.
- **6.3 Spill vs second card (RQ3)** — single-card spill (*prelim*: 27B Q4 ≈ 10 tok/s) vs dual (23 layer / 38 tensor). Cost-per-tok/s framing.
- **6.4 Model vs measurement (RQ4)** — roofline % achieved per config; where the gap comes from.
- **6.5 Energy (RQ5)** — J/token per config.
- **6.6 Backend comparison (secondary)** — Ollama vs llama.cpp, with Ollama's real placement stated.

**Figures:** decode vs size; prefill vs prompt length; link-BW sweep; roofline scatter; energy bars. All with error bars; colorblind-safe palette; plots generated by a script in-repo.

### 7. Discussion (~0.75 page)
- Practical guidance: decision table (model size × use case → config).
- Why the conventional wisdom fails at batch 1 and when it holds again.
- Generalization: what transfers to other consumer boards (x4 chipset slots, Gen5, 3+ GPUs) and what doesn't.

### 8. Threats to validity / Limitations (~0.5 page) — do not skip
- Single machine, single card sample per slot; vendor differences between cards (mitigated by slot-swap, Exp. A).
- Cloud reference systems differ in CPU, RAM, OS, virtualization and provider power limits; OS effect bounded by Exp. K, the rest stated, not controlled.
- Software churn: llama.cpp tensor-split mode is new and fast-moving; results pinned to one commit.
- Quantization level confounded with model size.
- Batch size 1 only (unless Exp. E).
- Power measurement limits (see M6).
- No quality evaluation; temperature-0 outputs of different lengths.

### 9. Conclusion (~0.25 page)
Restate finding + guidance + artifact link.

### Artifact availability statement
Zenodo DOI for code + raw logs + CSVs + analysis notebook/scripts; exact commit; how to regenerate every figure with one command.

### Appendix
Full per-tier tables, prompts verbatim, full CLI invocations, BIOS settings, sensitivity checks (flash-attn, mmap, `-ub`).

---

## Experiments to run

| ID | Experiment | Answers | Priority |
|---|---|---|---|
| A | **Slot swap** — move the MSI card to the x2 slot and the ASUS card to the x16 slot; rerun single0/single1 | Separates card vs slot effects | Must |
| B | **Link-bandwidth sweep** — force primary slot to Gen1/Gen2/Gen3 in BIOS (Gen1 x16 ≈ 4 GB/s ≈ current x2) and/or move GPU to other slots; rerun layer/tensor | RQ2 — the paper's novelty | Must |
| C | **Controlled microbenchmarks** with `llama-bench` (pp × tg × depth grid) | Clean phase separation, comparable to community numbers | Must |
| D | **PCIe traffic measurement** during runs (NVML `nvmlDeviceGetPcieThroughput` or `nvidia-smi dmon -s t`) | Validates the comm model | Should |
| E | **Batch/concurrency sweep** (1, 2, 4, 8 parallel requests via `--parallel`) | Where tensor split's advantage erodes | Should |
| F | **Prompt-length sweep** (512 → 8k) for prefill | Prefill link sensitivity | Should |
| G | Re-run existing tiers under the fixed protocol (M1–M10) | Final application-level numbers | Must |
| H | Sensitivity: flash-attn on/off, `-ub` sizes | Appendix robustness | Could |
| I | **Single 32 GB reference GPU** (cloud RTX 5090): same llama.cpp commit, same GGUFs, same harness | "Should I have bought one big card?" — tok/s, tok/s/$, tok/s/W | Should |
| J | **Symmetric dual-GPU box** (cloud, both GPUs on x16): layer vs tensor split | Symmetric-link control for RQ2 that can't be built locally | Should |
| K | **OS control**: rerun a small subset of the local rig under Linux (live USB) | Quantifies the Windows-vs-Linux confound introduced by cloud runs | Must if I/J run |

## Methodology work (harness changes before final data)

Status 2026-09-27: M1–M5, M7–M9, M11 implemented and smoke-tested on the rig. M6 largely
solved in software (both GPUs now expose NVML power + a hardware energy counter; a wall meter
would add whole-system energy). M10 partial: `summarize.py` produces CI tables; figure
scripts wait until the final figure set is decided.

Smoke-test observations worth keeping (Qwen3.5 9B Q4, n=2, not for citation):
- Under load GPU0 links at **PCIe 5.0 x8**, GPU1 at **PCIe 4.0 x2** — use these in §4 and the comm model.
- Tensor split during decode moves only ~170–350 MB/s over PCIe per GPU (<10% of the x2 link) → supports the latency-bound hypothesis.
- Tensor split *prefill* is 1.6–1.8× slower than single-GPU (1578 vs 2502 tok/s, chat tier) → the link-sensitive phase, as predicted.
- Decode energy/token is ~equal across single vs tensor (~2.5 J/tok): tensor split draws ~260 W vs ~178 W but is ~1.5× faster.

| ID | Fix | Why |
|---|---|---|
| M1 | Repetitions ≥ 5, report mean ± 95% CI (bootstrap) and CV | 2 runs can't support a claim |
| M2 | Disable prompt cache (`cache_prompt: false`) or perturb prompt per run | Run-2 prefill/TTFT currently warm-cache |
| M3 | Fixed output length (`ignore_eos` / `n_predict`) for throughput runs | Decode speed depends on generated length |
| M4 | Randomize config order; add cooldown + thermal steady-state check | Order/thermal bias |
| M5 | Record environment per run: driver, llama.cpp commit, Ollama version, clocks, temps, power limit, PCIe link gen/width (NVML) | Reproducibility; proves link state during Exp. B |
| M6 | Power: GPU0 (MSI) returns NVML NotSupported → add wall-power meter or validated alternative; otherwise drop energy claims | Current energy numbers cover one card only |
| M7 | Replace bandwidth metric with bytes-actually-read (active params for MoE) | Current metric exceeds physical peak for MoE |
| M8 | Record Ollama's real GPU placement; relabel config | Data shows Ollama uses both GPUs, not GPU0 |
| M9 | Log and report failures (crashes, OOM) as results, not silent gaps | e.g. Gemma 12B tensor split and Gemma 31B longdoc tensor split are missing — investigate |
| M10 | Single `make_figures.py` that regenerates every figure/table from raw CSVs | Artifact reproducibility |
| M11 | Make the harness cross-platform: paths/binaries via config or env vars, not hard-coded `D:\...\*.exe`; Ollama optional; record host ID per run | Required for cloud runs (Exp. I/J) and for others to reproduce |

---

## Best-practice checklist (tick before submission)

- [ ] Every number in the paper traces to a raw CSV row and a commit hash
- [ ] Error bars / CIs on every plotted point; n stated in every caption
- [ ] Hardware, software, and versions fully specified (someone could rebuild the rig)
- [ ] Prompts and exact CLI invocations in the appendix
- [ ] Negative/null results reported (e.g. layer split gives no speedup)
- [ ] Failures reported, not dropped
- [ ] Limitations section written honestly
- [ ] All citations verified against the source
- [ ] Code + data archived with DOI; license on both (code: MIT/Apache; data: CC-BY-4.0)
- [ ] `CITATION.cff` in repo
- [ ] AI-assistance disclosure per arXiv policy (author takes full responsibility for content)
- [ ] Figures readable in grayscale; colorblind-safe
- [ ] Abstract states concrete numbers, not "significant improvements"
- [ ] arXiv endorser lined up for cs.DC (or cs.PF)

## Sequencing

1. Harness fixes M1–M11 + finalize model list → 2. Exp. A + C + G (baseline) → 3. Exp. B + D (novelty) → 4. Analytical model fit
→ 5. Exp. I/J/K (cloud, batched into one rental window) → 6. Exp. E/F if time → 7. Figures script → 7. Write §4–6 → §5 → §7–8 → §2–3 → §1 → Abstract
→ 9. Zenodo release → 10. arXiv.

## Open questions

- Is GeForce P2P disabled on this platform (likely) — does the all-reduce go through host RAM? Determines the comm model.
- Does the B650 chipset uplink (shared x4) add contention with NVMe during model load? (Load-time only — probably out of scope.)
- Is the tensor-split speedup specific to llama.cpp's implementation? A second TP backend (e.g. vLLM on Windows/WSL, ExLlamaV2) would strengthen generality — worth it?
