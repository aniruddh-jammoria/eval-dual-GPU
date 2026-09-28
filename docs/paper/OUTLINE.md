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
is empirical: it measures where this holds and *where it breaks down* (prefill, long context,
batch > 1, narrower links) directly on hardware — including the inter-GPU PCIe traffic itself and a
controlled sweep of link bandwidth — and turns the results into guidance for building local inference rigs.

**Scope decision (2026-09-27):** no analytical/predictive modelling. Every claim is backed by a
measurement on a machine; explanations are stated only where a measurement shows them (e.g. PCIe
traffic per phase, the link-bandwidth sweep).

---

## Research questions

| # | Question | Why it matters |
|---|---|---|
| RQ1 | How does split strategy (single / layer / tensor) affect decode and prefill throughput across model sizes and architectures (dense vs MoE)? | Main result |
| RQ2 | How sensitive is each strategy to inter-GPU link bandwidth? | The novel part — needs the link-bandwidth sweep (Exp. B) |
| RQ3 | What is the cost of VRAM overflow on one card vs adding a second card? | Practical "buy a second GPU?" question |
| RQ4 (secondary) | Energy per token and cost per tok/s across configs | Practitioner value |

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
  3. Direct measurement of inter-GPU PCIe traffic per inference phase, showing which phase the link constrains.
  4. Open dataset, harness, and dashboard.
- **Figure 1** (the "money plot"): decode tok/s vs model size, one line per config, spill region shaded.

**To do:** nothing until results are final; draft last-but-one.

### 2. Background (~0.75 page)
**Contents**
- LLM inference phases: prefill (processes the whole prompt at once) vs decode (one token at a time, reads all active weights per token).
- Parallelism strategies: layer/pipeline split (sequential GPUs, activations passed once per boundary) vs tensor split (Megatron-style, all-reduce per layer, both GPUs active).
- Consumer PCIe topology: CPU lanes vs chipset lanes, shared chipset uplink (x4 Gen4 on B650), no P2P on GeForce → transfers bounce through host memory.
- Quantized GGUF formats; MoE active-parameter reads.
- **Figure 2:** system topology diagram (CPU, chipset, two GPUs, link widths).

**To do:** verify whether P2P is available on this platform (`nvidia-smi topo -m`, CUDA `p2pBandwidthLatencyTest`) — determines how inter-GPU traffic flows; report it in §4.

### 3. Related work (~0.5–0.75 page)
**Contents** — group by theme, position this paper in one sentence per group:
- Parallelism for LLMs: Megatron-LM (tensor), GPipe/PipeDream (pipeline).
- Inference systems & offloading on constrained hardware: FlexGen, PowerInfer, llama.cpp, vLLM, ExLlama.
- Benchmarking methodology: MLPerf Inference; LLM-perf leaderboards.
- Community benchmarks of consumer multi-GPU (llama.cpp discussions, blog posts) — cite as grey literature, note lack of controlled topology.

**To do:** literature pass; build `references.bib`; verify every citation exists and says what we claim (no citing from memory).

### 4. Experimental setup (~1 page)
**Contents**
- Hardware table (CPU, board, RAM, both GPUs with slot/link/vendor), driver version, power limits, clocks, cooling/ambient.
- Software: OS build, CUDA, llama.cpp commit hash, Ollama version, Python harness commit.
- Models table: params, active params (MoE), quant, GGUF bytes, layers, hidden dim.
- Model selection rationale (target ~12–15 models, chosen to de-confound, not to maximize count):
  - size ladder within one family at a fixed quant (Q4_K_M) — clean size axis for Fig. 1
  - dense 2–3 models with GGUF 13–18 GB — the fits-vs-spills boundary on one card
  - matched dense/MoE pairs at similar total size — tests the MoE tensor-split effect
  - 1–2 additional families (different hidden dim / layer count) — varies how much data tensor split exchanges per layer
  - one mid-size model at Q4/Q6/Q8 — isolates bytes-per-token from architecture
- Reference systems (cloud, see Exp. I/J): single 32 GB GPU (RTX 5090) and a symmetric dual-GPU box (both x16). Framed as reference points, compared on tok/s, tok/s per $, tok/s per W — not as controlled like-for-like.
- Configurations table: single0, single1, layer split, tensor split, Ollama (with *actual* observed placement).
- Workloads: controlled microbenchmarks (`llama-bench`: pp512/pp2048, tg128 at depths 0/2k/4k) **plus** application-level tiers (chat/rag/longdoc/code via server). Justify both.
- Metrics with exact definitions and sources (decode tok/s, prefill tok/s, TTFT, energy/token, peak VRAM, host RSS, PCIe throughput).
- Measurement protocol: warm-up, ≥5 repetitions, randomized config order, cache disabled, fixed output length, thermal steady-state check, idle-system checks.
- **Table 1–3:** hardware/software, models, configs.

**To do:** see Methodology work list (M1–M10) below.

### 5. Results (~3 pages)
One subsection per RQ, each opening with the takeaway sentence.
- **5.1 Split strategy vs model size (RQ1)** — Fig. 1 + table of decode/prefill per model×config with 95% CIs. Dense vs MoE contrast (*prelim*: MoE gains less from tensor split, ~+17% vs ~+50–65% dense).
- **5.2 Sensitivity to link bandwidth (RQ2)** — Exp. B: throughput vs link BW (Gen5 x8 → Gen1 / x2), per strategy and phase, plus measured PCIe traffic per phase (Exp. D). Hypothesis to test: decode flat, prefill slows.
- **5.3 Spill vs second card (RQ3)** — single-card spill (*prelim*: 27B Q4 ≈ 10 tok/s) vs dual (23 layer / 38 tensor). Cost-per-tok/s framing.
- **5.4 Energy (RQ4)** — J/token per config.
- **5.5 Backend comparison (secondary)** — Ollama vs llama.cpp, with Ollama's real placement stated.

**Figures:** decode vs size; prefill vs prompt length; link-BW sweep; PCIe traffic per phase; energy bars. All with error bars; colorblind-safe palette; plots generated by a script in-repo.

### 6. Discussion (~0.75 page)
- Practical guidance: decision table (model size × use case → config).
- Why the conventional wisdom fails at batch 1 and when it holds again.
- Generalization: what transfers to other consumer boards (x4 chipset slots, Gen5, 3+ GPUs) and what doesn't.

### 7. Threats to validity / Limitations (~0.5 page) — do not skip
- Single machine, single card sample per slot; vendor differences between cards (mitigated by slot-swap, Exp. A).
- Cloud reference systems differ in CPU, RAM, OS, virtualization and provider power limits; OS effect bounded by Exp. K, the rest stated, not controlled.
- Software churn: llama.cpp tensor-split mode is new and fast-moving; results pinned to one commit.
- Quantization level confounded with model size.
- Batch size 1 only (unless Exp. E).
- Power measurement limits (see M6).
- No quality evaluation; temperature-0 outputs of different lengths.

### 8. Conclusion (~0.25 page)
Restate finding + guidance + artifact link.

### Artifact availability statement
Zenodo DOI for code + raw logs + CSVs + analysis notebook/scripts; exact commit; how to regenerate every figure with one command.

### Appendix
Full per-tier tables, prompts verbatim, full CLI invocations, BIOS settings, sensitivity checks (flash-attn, mmap, `-ub`).

---

## Experiments to run

### Reference configuration (fixed for all core experiments)

Split mode is the independent variable; everything else is pinned here. Configuration
experiments (X-series) change **one** of these settings at a time. A setting is promoted into
the reference only if it changes a conclusion — then core runs are redone once.

| Setting | Reference value | Notes |
|---|---|---|
| llama.cpp | b11146 (v0.5.0), commit 7fe450e19, Windows CUDA 13.4 build | frozen; no NCCL in Windows build (see X2) |
| NVIDIA driver / Ollama | 591.86 / 0.34.1 | frozen |
| Split modes (independent variable) | `none` (single0, single1), `layer` (dual), `tensor` (dual_tensor) | `row` is deprecated upstream — only as X6 fallback |
| GPU offload | `-ngl -1` (all layers) | spill cases handled by llama.cpp `--fit` default |
| Context | `-c 8192` (sweep: longest prompt + gen, rounded to 256) | |
| Batch sizes | `-ub 512` (upstream default), `-b 2048` | changed from `-ub 2048` after X1 (2026-09-29): 512 gives the best layer-split prefill and is within ~1% of the best for single and tensor. Pre-X1 data ran `-ub 2048` |
| Flash attention | on | required by tensor mode |
| KV cache type | f16 | tensor mode supports only non-quantized KV |
| mmap | on | |
| Resizable BAR | on for both GPUs (BAR1 = 16 GiB) | recorded per session (`bar1_total_mib`) |
| Windows VBS / IOMMU | VBS running (memory integrity), hypervisor present | uses the IOMMU; state in §4; Linux run (K) will differ |
| P2P | unavailable — `cudaDeviceCanAccessPeer` = 0 both ways (2026-09-29) | all inter-GPU traffic goes via host RAM; see X3 |
| Tensor split ratio | `1,1` | |
| Sampling | temperature 0, seed 0, `ignore_eos`, `cache_prompt=false` | measurement protocol, not a tuning choice |

### Core experiments (answer the research questions)

| ID | Experiment | Answers | Priority | Status |
|---|---|---|---|---|
| A | **Slot swap** — move the MSI card to the x2 slot and the ASUS card to the x8 slot; rerun single0/single1 | Separates card vs slot effects | Must | needs you (hardware) |
| B | **Link-bandwidth sweep** — force primary slot to Gen1/Gen2/Gen3/Gen4 in BIOS and/or move GPU to other slots; rerun layer/tensor | RQ2 — the paper's novelty | Must | blocked on BIOS check |
| C | **Controlled microbenchmarks** with `llama-bench` (pp × tg × depth grid) | Clean phase separation, comparable to community numbers | Should | not started |
| D | **PCIe traffic measurement** per phase (NVML throughput, recorded every repetition) | Shows directly how much data crosses the link in each phase | Must | built into harness |
| E | **Batch/concurrency sweep** (1, 2, 4, 8 parallel requests via `--parallel`) | Where tensor split's advantage erodes | Should | not started |
| F | **Prompt-length sweep** (128 → 8192 tokens), `run.py sweep` | Prefill/TTFT scaling per split mode | Must | 9B pilot done 2026-09-28; rest with final model list |
| G | **Application tiers** (chat/rag/longdoc/code) under the final protocol, all models | Final application-level numbers | Must | waits on model list |
| I | **Single 32 GB reference GPU** (cloud RTX 5090): same llama.cpp commit, same GGUFs, same harness | "Should I have bought one big card?" — tok/s, tok/s/$, tok/s/W | Should | later |
| J | **Symmetric dual-GPU box** (cloud, both GPUs on x16): layer vs tensor split | Symmetric-link control for RQ2 that can't be built locally | Should | later |
| K | **OS control**: rerun a subset of the local rig under Linux (live USB), same build number | Quantifies the Windows-vs-Linux confound; hosts X2 | Must | needs Linux boot media |

### Configuration experiments (one setting changed from the reference)

Run on a 3-model subset — one small dense (Qwen3.5 9B Q4), one large dense that spills on one
card, one MoE — with the split modes that the setting can affect. Report as sensitivity results;
promote into the reference only if a conclusion changes.

| ID | Setting changed | Values | Why | Priority |
|---|---|---|---|---|
| X1 | Prompt batch sizes `-ub` (and `-b ≥ -ub`) | 512, 1024, 2048, 4096 | **Done 2026-09-29 (9B, n=5): changed a conclusion → reference promoted to `-ub 512`.** See results below | Done |
| X2 | **NCCL** (Linux build with NCCL vs Windows build without) | on/off, tensor split only | Upstream docs: NCCL "provides necessary optimizations for tensor mode performance"; the Windows build has none, so our tensor-split numbers may understate the hardware. Runs inside Exp. K | Must |
| X3 | GPU peer-to-peer `GGML_CUDA_P2P=1` | set/unset | Direct GPU↔GPU transfers instead of via host RAM. **Done 2026-09-29: P2P unavailable** (`cudaDeviceCanAccessPeer` = 0 both directions, driver 591.86, Windows with VBS running) — `GGML_CUDA_P2P` has nothing to enable. Re-check under Linux in Exp. K | Done |
| X4 | Single-GPU spill placement for MoE: `--n-cpu-moe N` / `-ot` experts→CPU vs default | default vs experts-on-CPU | Fair baseline for RQ3 ("spill vs buy a second card"): reviewers will ask whether single-GPU was run the recommended way | Must for RQ3 |
| X5 | Second tensor-parallel implementation: `ik_llama.cpp` "split mode graph" | llama.cpp tensor vs ik_llama.cpp graph | Tests whether the tensor-split pattern is general or implementation-specific (replaces the old open question) | Could |
| X6 | `--split-mode row` | row vs tensor | Deprecated upstream and reported slow on PCIe — only as a workaround for Gemma 12B, whose `tensor` mode crashes (GGML_ASSERT in ggml-backend-meta.cpp, b9858 and b11146) | Could |
| H | Flash attention off, mmap off | on/off each | Appendix robustness | Could |

Out of scope for configuration experiments (decided 2026-09-29): speculative decoding (mixed
results on consumer GPUs, including a slowdown on RTX 5060 Ti), quantized KV cache (unsupported
in tensor mode), prompt caching / slot reuse (we disable caching to measure cleanly; separate question).

## Methodology work (harness changes before final data)

Status 2026-09-27: M1–M5, M7–M9, M11 implemented and smoke-tested on the rig. M6 largely
solved in software (both GPUs now expose NVML power + a hardware energy counter; a wall meter
would add whole-system energy). M10 partial: `summarize.py` produces CI tables; figure
scripts wait until the final figure set is decided.

Smoke-test observations worth keeping (Qwen3.5 9B Q4, n=2, not for citation):
- Under load GPU0 links at **PCIe 5.0 x8**, GPU1 at **PCIe 4.0 x2** — use these in §4.
- Tensor split during decode moves only ~170–350 MB/s over PCIe per GPU (<10% of the x2 link) → supports the latency-bound hypothesis.
- Tensor split *prefill* is 1.6–1.8× slower than single-GPU (1578 vs 2502 tok/s, chat tier) → likely the link-sensitive phase; Exp. B/D will confirm.
- Decode energy/token is ~equal across single vs tensor (~2.5 J/tok): tensor split draws ~260 W vs ~178 W but is ~1.5× faster.

Prompt-length sweep, Qwen3.5 9B Q4, llama.cpp b11146, n=5, 128–8192 tokens (2026-09-28, `results/runs/2026-09-28_23-40-11_sweep_*`):
- Tensor-split prefill is a **roughly constant ~0.6–0.67× of single-GPU at every length** (worst at 128 tokens: 0.58×). The earlier idea that the penalty *grows* with prompt length is **not** supported. Whether the penalty comes from the x2 link has to be settled by Exp. B.
- Tensor-split decode stays at **1.47–1.50×** single-GPU from 128 to 8192 tokens of context.
- Layer split matches single-GPU prefill up to 4096 tokens, then is **1.24× faster at 8192** — likely pipelining across GPUs once a prompt spans several 2048-token micro-batches (`-ub`). Testable with a `-ub` sweep (already in Ideas.md).
- Mean PCIe RX on GPU1 during tensor split rises with prompt length (262 → 1370 MB/s averaged over the whole request).
- From these numbers, tensor split beats single-GPU in total request time once the answer exceeds ~11 tokens (128-token prompt) or ~290 tokens (8192-token prompt).

Exp. X1 — `-ub` sweep, Qwen3.5 9B Q4, b11146, n=5, prompts 512–8192, `-ub` 512/1024/2048/4096 (2026-09-29, 240 reps, 0 failures, `results/runs/2026-09-29_00-*_sweep_*`):
- **Layer-split prefill speedup depends on how many `-ub` chunks the prompt spans**, and only on that: 1 chunk ≈ 0.94–0.99× single, 2 chunks ≈ 0.96–1.01×, 4 chunks ≈ 1.21–1.25×, 8 chunks ≈ 1.41–1.43×, 16 chunks = 1.55×. Consistent across every (length, `-ub`) pair — i.e. llama.cpp pipelines successive chunks across the two GPUs.
- With `-ub 512`, **layer split prefill beats single-GPU by 22–55% for prompts ≥ 2048 tokens** (8192: 4421 vs 2845 tok/s) — the pre-X1 conclusion "layer split ≈ single for prefill" was an artefact of `-ub 2048`.
- **Tensor-split prefill is insensitive to `-ub`** (0.62–0.67× single at every setting) — its penalty is not a chunking artefact.
- Single-GPU prefill varies ≤ ~4% with `-ub` (best at 1024); decode is unaffected by `-ub` in every mode.
- Larger `-ub` costs VRAM: single-GPU peak 8.0 GiB at 512 → 8.8 GiB at 4096 (8192-token prompt).
- Consequence at `-ub 512`, 8192-token prompt: layer split has the best TTFT (1.85 s vs 2.88 single, 4.36 tensor); tensor split has the best decode (97 vs 65–67 tok/s) and wins total request time only for answers > ~500 tokens.

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

1. Harness fixes M1–M11 ✓, llama.cpp frozen ✓, X3 ✓, finalize model list → 2. X1 (cheap, may change the reference)
→ 3. Exp. F + G on all models (baseline) + A → 4. Exp. B + X4 (novelty, RQ3 fairness)
→ 5. Exp. K + X2 (Linux/NCCL) → 6. Exp. I/J (cloud, one rental window) → 7. Exp. E, X5 if time → 8. Figures script → 9. Write §4–5 → §6–7 → §2–3 → §1 → Abstract
→ 10. Zenodo release → 11. arXiv.

## Open questions

- ~~Is GeForce P2P disabled on this platform?~~ Yes (X3): inter-GPU traffic goes through host RAM. Report in §4; re-check on Linux.
- Does the B650 chipset uplink (shared x4) add contention with NVMe during model load? (Load-time only — probably out of scope.)
- Is the tensor-split speedup specific to llama.cpp's implementation? → X5 (ik_llama.cpp); vLLM/ExLlamaV2 only if X5 is inconclusive.

---

## To-do — for you (things only you can do)

Updated 2026-09-29. Tick as you go; Claude handles everything not on this list.

**Now (unblocks the next experiments)**
- [ ] **BIOS check (Exp. B):** can PCI_E1's PCIe generation be set (Gen1–Gen5)? Del at boot → F7 (Advanced) → Settings → Advanced → PCI Subsystem Settings; look for e.g. "PCI_E1 Gen Switch" / "PCIe Link Speed". Photo of the options. (Resizable BAR, IOMMU/VBS and P2P already measured from Windows — nothing to check for those.)
- [ ] **Model list:** tell Claude which model families/sizes you're interested in and have bandwidth to download; Claude maps them to the selection criteria in §4, then you download the agreed GGUFs into `D:\llama-models`.
- [ ] **Freeze software:** turn off automatic updates for the NVIDIA driver (NVIDIA App), Ollama, and Windows Update driver installs until data collection is done. Frozen: llama.cpp b11146, driver 591.86, Ollama 0.34.1.
- [ ] **Keep the PC idle during runs** (no games/video/other GPU work); runs will be announced with an estimated duration.

**Soon**
- [ ] **Slot swap (Exp. A):** physically swap the two cards between slots for one run session, then swap back. Claude will tell you when.
- [ ] **Linux boot media (Exp. K, X2):** make an Ubuntu 24.04+ live USB or dual-boot so the same rig can run Linux with NCCL.
- [ ] Optional: a wall-power meter (whole-system energy) — nice to have, not required.

**Later (after results)**
- [ ] Decide cloud budget for Exp. I/J (single RTX 5090 + a symmetric dual-GPU box), and create an account with a GPU rental provider.
- [ ] Choose licenses (e.g. MIT for code, CC-BY-4.0 for data).
- [ ] Check the author name in `CITATION.cff`.
- [ ] Enable Zenodo's GitHub integration, then publish GitHub releases for the tags (release notes drafted in chat 2026-09-27).
- [ ] Find an arXiv endorser for cs.DC (or cs.PF).
