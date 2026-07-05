# Dual RTX 5060 Ti 16 GB — LLM Inference Benchmark
(Template file for reports)

Date run: DD-MMM-YYYY

---

### Hardware specifications
**GPU0:** MSI VENTUS 2X OC (PCIe 4.0 x16, from CPU — 32 GB/s) — primary  
**GPU1:** ASUS DUAL OC (PCIe 4.0 x2, from B650 chipset — 4 GB/s)  
**Motherboard:** XX
**CPU:** XX
**Hardware:** 2× NVIDIA RTX 5060 Ti 16 GB  

### Software specifications
**Ollama:** Ollama version XXX 
**llama.cpp:** llama.cpp bXXX (CUDA xx.x)  

---

### Inference configurations
Create a table which outlines all inference configurations + backends
| Label | Backend | GPU config | Notes |
|---|---|---|---|
Label: Provide a label that is used consistently throghout the report; use abbreviations consistently.
Backend: Ollama or Llama.cpp or vLLM etc
GPU config: Single or Dual etc.
Notes: Any other extra config parameters eg split tensor. Explain in 1-line if too technical. 

---

### Response configurations
Create a table which outlines all response tiers
- Tier name
- Input # of tokens
- Output # of tokens
- Relevance (why we use this tier, in 1 line)
- Task description, 1 line

---

### Metrics
Create a table which defines all metrics
- Metric name
- Definition
- Formula
- Calculated from/via > mention where these metrics come from

---
### Prefill and Decode token generation
(Repeat the following for all reponse tiers; use information from the latest run)

#### Response tier name — decode tok/s (input: ~540 tok, max output: 512 tok)
Create a table which contains Prefill/Prompt tok/s for all configurations
- Columns are: Model name (use official name), Model size (GB), Decode tok/s, Prefill tok/s, TTFs
- Decode tok/s, Prefill tok/s and TTFS columns are further split into sub-columns, one column for each configuration.
- Rows are for models - each model variant gets its own row
- Place models from the same provider together (eg all Gemma and Qwen models together)
- Within a certain provider, list models in ascending order of size

---

### Peak energy consumption
Createe one consolidated table for all runs
- Columns are: Model name (use official name), Model size (GB) + one column for each configuration
- Rows are for models - each model variant gets its own row
- Each cell contains peak energy consumption for the corresponding model + configuration combination. Mention a range across all response tiers

### Key Observations
Write down key observation for the following themes. Each theme gets its own section heading (max 1 sentence, descriptive), and can have multiple bullet points. Be data-oriented and concise, use information only from the tables above. Include a recommendation, if the data supports it.
- Impact of model size on metrics
- Impact of PCIe slot on metrics
- Impact of backend on metrics
- Impact of single vs dual GPU on metrics
- Impact of special configs (eg tensor split)

### General guidance for writing the template (do not include in the final report)
- Use abbreviations consistently, update the below if needed
    - OL: Ollama
    - LC: Llama.cpp
- Use the latest run results (from results/)
- Update XXXs with actual values
- File title should be benchmark_results_yyyymmdd.md