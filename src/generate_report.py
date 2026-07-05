"""
Generate docs/index.html from results/metrics/*.csv

Usage:  python generate_report.py
Output: docs/index.html  (ready for GitHub Pages)

Config:
  RUNCOUNT  — number of latest runs to average per (model,tier,backend,gpu) key.
              Run-count badge in the dashboard turns amber when actual < RUNCOUNT.
"""

import csv, json, re, sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent   # repo root (src/ lives under it)
METRICS_DIR = ROOT / "results" / "metrics"
DOCS_DIR    = ROOT / "docs"

# ── Config ────────────────────────────────────────────────────────────────────
RUNCOUNT = 2   # latest N sessions to average per cell key

MODEL_ORDER = [
    "gemma4-12b-qat",
    "qwen3.5-9b-q4",
    "qwen3.5-9b-q8",
    "qwen3.6-27b-q4",
    "qwen3.6-27b-q6",
    "gemma4-26b-moe-q4",
    "qwen3.6-35b-a3b-q4",
    "gemma4-31b-q5",
]

TIER_META = {
    "chat":    ("~540 tok in · 512 max out",
                "Dracula Ch. I (400 words) — summarise Harker's first impressions of Eastern Europe"),
    "rag":     ("~2 000 tok in · 1 024 max out",
                "Dracula Ch. I (1 600 words) — list every warning or sign of danger before Castle Dracula"),
    "longdoc": ("~4 000 tok in · 1 024 max out",
                "Dracula Ch. I (3 200 words) — detailed atmospheric and supernatural analysis"),
    "code":    ("~155 tok in · 1 024 max out",
                "Implement Python RateLimiter — sliding-window, decorator, context manager, thread-safe"),
}

NUMERIC_FIELDS = [
    "decode_tok_s", "prompt_tok_s", "ttft_s",
    "bw_gb_s", "bw_pct", "gpu0_gib", "gpu1_gib",
    "avg_watts", "peak_watts",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_model(s):
    parts = s.split("  —  ", 1)
    mid = parts[0].strip()
    if len(parts) < 2:
        return mid, mid, ""
    full = parts[1].strip()
    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", full)
    return (mid, m.group(1).strip(), m.group(2).strip()) if m else (mid, full, "")

def is_moe(mid):
    return any(x in mid for x in ("moe", "a3b", "a4b"))

def is_spill(cfg, gpu0, gpu1):
    return (cfg == "single0" and gpu0 > 14.5) or (cfg == "single1" and gpu1 > 14.5)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    """Read all metric CSVs; for each (model,tier,backend,gpu) key keep the
    latest RUNCOUNT rows and return their averaged numeric fields."""
    all_rows: dict = {}
    csvs = sorted(f for f in METRICS_DIR.glob("*.csv") if f.name != "results.csv")
    if not csvs:
        sys.exit(f"No CSV files found in {METRICS_DIR}. Run: python run.py run-all")

    for path in csvs:
        for row in csv.DictReader(open(path, encoding="utf-8")):
            if not row.get("decode_tok_s"):
                continue
            mid, display, quant = parse_model(row["model"])
            key = (mid, row["tier"], row["backend"], row["gpu_config"])
            all_rows.setdefault(key, []).append(
                {**row, "mid": mid, "display": display, "quant": quant}
            )

    result = {}
    for key, rlist in all_rows.items():
        recent = rlist[-RUNCOUNT:]
        merged = {**recent[-1], "_runs": len(recent)}
        for field in NUMERIC_FIELDS:
            vals = []
            for r in recent:
                v = r.get(field)
                if v not in (None, "", "None"):
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            merged[field] = sum(vals) / len(vals) if vals else None
        result[key] = merged
    return result

# ── Tier / cell builder ───────────────────────────────────────────────────────
def build_tiers(rows):
    out = {}
    for tier, (desc, prompt) in TIER_META.items():
        tier_rows = []
        for mid in MODEL_ORDER:
            any_row = next((v for k, v in rows.items() if k[0] == mid and k[1] == tier), None)
            if not any_row:
                continue

            # estimate GGUF size from bw ÷ decode
            gguf = None
            for cfg_try in ("single0", "single1", "ollama"):
                be  = "ollama"   if cfg_try == "ollama" else "llamacpp"
                gcf = "auto"     if cfg_try == "ollama" else cfg_try
                r = rows.get((mid, tier, be, gcf))
                if r and r.get("bw_gb_s") and r.get("decode_tok_s"):
                    try:
                        gguf = round(float(r["bw_gb_s"]) / float(r["decode_tok_s"]), 1)
                        break
                    except Exception:
                        pass

            def cell(be, cfg, _mid=mid, _tier=tier):
                r = rows.get((_mid, _tier, be, cfg))
                if not r:
                    return {"decode": None, "prefill": None, "ttft": None,
                            "bw": None, "watts": None, "spill": False, "runs": 0}
                g0 = float(r.get("gpu0_gib") or 0)
                g1 = float(r.get("gpu1_gib") or 0)
                spill = is_spill(cfg, g0, g1)

                def fv(k, digits=1):
                    v = r.get(k)
                    if v is None:
                        return None
                    try:
                        f = float(v)
                    except (ValueError, TypeError):
                        return None
                    # treat 0 as no-data for power (pynvml not installed)
                    if f <= 0 and k in ("avg_watts", "peak_watts"):
                        return None
                    return round(f, digits)

                return {
                    "decode":  fv("decode_tok_s"),
                    "prefill": fv("prompt_tok_s", 0),
                    "ttft":    fv("ttft_s", 3),
                    "bw":      fv("bw_gb_s"),
                    "watts":   fv("avg_watts"),
                    "spill":   spill,
                    "runs":    r.get("_runs", 1),
                }

            tier_rows.append({
                "id":     mid,
                "model":  any_row["display"],
                "quant":  any_row["quant"],
                "gguf":   gguf,
                "moe":    is_moe(mid),
                "ollama": cell("ollama",   "auto"),
                "gpu0":   cell("llamacpp", "single0"),
                "gpu1":   cell("llamacpp", "single1"),
                "dual":   cell("llamacpp", "dual"),
                "tensor": cell("llamacpp", "dual_tensor"),
            })

        if tier_rows:
            out[tier] = {"desc": desc, "prompt": prompt, "rows": tier_rows}
    return out

# ── Stats ─────────────────────────────────────────────────────────────────────
def compute_global_max(tiers):
    maxes = {"decode": 0.0, "prefill": 0.0, "ttft": 0.001, "bw": 0.0, "watts": 1.0}
    for tier_data in tiers.values():
        for row in tier_data["rows"]:
            for col in ("ollama", "gpu0", "gpu1", "dual", "tensor"):
                d = row.get(col) or {}
                for k in maxes:
                    v = d.get(k)
                    if v and v > maxes[k]:
                        maxes[k] = v
    return maxes

def compute_stats(tiers):
    peak_tps, peak_model = 0.0, ""
    best_lift, lift_desc = 0.0, ""
    models_seen = set()
    for tier_data in tiers.values():
        for row in tier_data["rows"]:
            models_seen.add(row["id"])
            for col in ("ollama", "gpu0", "gpu1", "dual", "tensor"):
                v = (row.get(col) or {}).get("decode")
                if v and v > peak_tps:
                    peak_tps = v
                    peak_model = f"{row['model']}, {col}"
            single = (row.get("gpu0") or {}).get("decode")
            tensor = (row.get("tensor") or {}).get("decode")
            if single and tensor and single > 0:
                lift = tensor / single
                if lift > best_lift:
                    best_lift = lift
                    lift_desc = f"{row['model']}: {single:.1f} → {tensor:.1f} tok/s"
    return peak_tps, peak_model, best_lift, lift_desc, len(models_seen)

# ── HTML template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = '''\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTX 5060 Ti × 2 — LLM Benchmark</title>
<style>
:root{{
  --bg:#0c0e14;--surf:#131620;--card:#191c28;--border:#23263a;
  --text:#cdd0e0;--muted:#535770;--accent:#00c98f;--amber:#f0a030;
  --mono:'SF Mono','Fira Code','Cascadia Code','Consolas',monospace;
  --sans:system-ui,-apple-system,'Segoe UI',sans-serif;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased}}
a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}
.wrap{{max-width:1160px;margin:0 auto;padding:0 24px}}
header{{border-bottom:1px solid var(--border);padding:32px 0 28px}}
.header-inner{{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;flex-wrap:wrap}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:6px}}
h1{{font-family:var(--mono);font-size:clamp(20px,4vw,30px);font-weight:600;color:#edf0ff;letter-spacing:-.02em;line-height:1.2}}
h1 span{{color:var(--muted);font-weight:400}}
.hw-chips{{display:flex;flex-wrap:wrap;gap:6px;margin-top:16px}}
.chip{{font-family:var(--mono);font-size:11px;padding:3px 9px;border:1px solid var(--border);border-radius:3px;color:var(--muted);background:var(--surf);white-space:nowrap}}
.chip b{{color:var(--text);font-weight:600}}
.chip.gpu0{{border-color:#2a3a60}}.chip.gpu1{{border-color:#3a2a60}}
.meta-right{{text-align:right;flex-shrink:0}}
.meta-date{{font-family:var(--mono);font-size:12px;color:var(--muted)}}
.meta-sw{{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:4px}}
.meta-sw span{{color:var(--text)}}
/* stat tiles */
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin:28px 0}}
.stat{{background:var(--card);padding:20px 22px}}
.stat-label{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}}
.stat-value{{font-family:var(--mono);font-size:28px;font-weight:700;color:var(--accent);line-height:1;font-variant-numeric:tabular-nums}}
.stat-value.dim{{color:var(--text);font-size:24px}}
.stat-sub{{font-size:11px;color:var(--muted);margin-top:4px}}
/* metric selector */
.metric-bar{{display:flex;align-items:center;gap:6px;padding:20px 0 14px;flex-wrap:wrap}}
.metric-bar-lbl{{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-right:6px;white-space:nowrap}}
.mpill{{font-family:var(--mono);font-size:11px;padding:5px 13px;border-radius:20px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;transition:color .12s,border-color .12s,background .12s;white-space:nowrap;line-height:1}}
.mpill:hover{{color:var(--text);border-color:color-mix(in srgb,var(--muted) 70%,transparent)}}
.mpill.active{{background:var(--accent);color:#080a0f;border-color:var(--accent);font-weight:700}}
.mpill:disabled{{opacity:.35;cursor:not-allowed;pointer-events:none}}
.mpill-unit{{font-size:9px;opacity:.65;font-weight:400}}
.metric-hint{{font-family:var(--mono);font-size:10px;color:var(--muted);padding:0 0 10px;letter-spacing:.02em}}
.metric-hint b{{color:var(--text)}}
/* tier tabs */
.tabs{{display:flex;border-bottom:1px solid var(--border)}}
.tab{{font-family:var(--mono);font-size:12px;padding:10px 20px;cursor:pointer;border:none;background:none;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-1px;transition:color .15s,border-color .15s;white-space:nowrap}}
.tab:hover{{color:var(--text)}}.tab.active{{color:var(--accent);border-bottom-color:var(--accent)}}
.tier-meta{{font-size:10px;color:var(--muted);display:block;margin-top:1px}}
.tab.active .tier-meta{{color:color-mix(in srgb,var(--accent) 50%,var(--muted))}}
.table-prompt{{font-size:12px;color:var(--muted);padding:10px 0 14px;font-style:italic}}
.table-prompt b{{color:var(--text);font-style:normal;font-weight:600}}
/* table */
.table-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
thead th{{padding:8px 14px;text-align:right;font-family:var(--mono);font-size:11px;letter-spacing:.06em;color:var(--muted);background:var(--surf);border-bottom:1px solid var(--border);white-space:nowrap}}
thead th.col-model{{text-align:left}}
thead th.col-best{{color:var(--accent)}}
.th-sub{{display:block;font-size:9px;color:var(--border);margin-top:1px;letter-spacing:.04em}}
thead th.col-best .th-sub{{color:color-mix(in srgb,var(--accent) 40%,transparent)}}
tbody tr{{border-bottom:1px solid var(--border)}}
tbody tr:last-child{{border-bottom:none}}
tbody tr:hover td{{filter:brightness(1.08)}}
td{{padding:0;vertical-align:top}}
td.col-model{{padding:10px 14px;min-width:200px;vertical-align:middle}}
.model-name{{font-family:var(--mono);font-size:13px;color:var(--text);font-weight:600}}
.model-badges{{display:flex;gap:5px;margin-top:4px;flex-wrap:wrap}}
.badge{{font-family:var(--mono);font-size:10px;padding:1px 6px;border-radius:2px;background:var(--surf);border:1px solid var(--border);color:var(--muted);letter-spacing:.04em}}
.badge.moe{{border-color:#3a2855;color:#a070d0;background:#1a1020}}
td.col-val{{padding:9px 14px 4px;text-align:right;min-width:90px}}
.cell-inner{{font-family:var(--mono);font-size:14px;font-weight:600;display:flex;align-items:center;justify-content:flex-end;gap:4px}}
.cell-inner.best{{font-size:15px}}
.spill-mark{{font-size:10px;opacity:.8}}
.cell-dash{{color:var(--muted);font-size:13px}}
/* run count badge */
.run-ct{{font-family:var(--mono);font-size:8px;text-align:right;color:var(--muted);opacity:.45;letter-spacing:.04em;padding-bottom:3px}}
.run-ct.stale{{color:var(--amber);opacity:.75}}
/* legend + findings */
.legend{{display:flex;gap:20px;flex-wrap:wrap;padding:10px 0 0;font-size:11px;color:var(--muted);border-top:1px solid var(--border)}}
.legend-item{{display:flex;align-items:center;gap:5px}}
.legend-swatch{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
.findings{{margin:40px 0 0}}
.section-title{{font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);padding-bottom:12px;border-bottom:1px solid var(--border)}}
.finding-list{{list-style:none;display:flex;flex-direction:column}}
.finding{{display:grid;grid-template-columns:auto 1fr;gap:0 20px;padding:18px 0;border-bottom:1px solid var(--border);align-items:start}}
.finding:last-child{{border-bottom:none}}
.finding-label{{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding-top:2px;white-space:nowrap;min-width:100px}}
.finding-body{{font-size:13px;line-height:1.65;color:var(--text)}}
.finding-body strong{{color:#edf0ff;font-weight:600}}
.num{{font-family:var(--mono);color:var(--accent);font-size:13px}}
.num-amber{{font-family:var(--mono);color:var(--amber);font-size:13px}}
footer{{border-top:1px solid var(--border);margin-top:48px;padding:20px 0;font-size:11px;color:var(--muted)}}
.footer-inner{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}}
footer a{{color:var(--muted)}}footer a:hover{{color:var(--text)}}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="header-inner">
      <div>
        <div class="eyebrow">LLM Inference Benchmark</div>
        <h1>Dual RTX 5060 Ti <span>/ 16 GB</span></h1>
        <div class="hw-chips">
          <span class="chip gpu0"><b>GPU0</b> MSI VENTUS 2X OC &middot; PCIe 4.0 x16 (CPU)</span>
          <span class="chip gpu1"><b>GPU1</b> ASUS DUAL OC &middot; PCIe 4.0 x2 (chipset)</span>
          <span class="chip"><b>CPU</b> AMD Ryzen 9 7900</span>
          <span class="chip"><b>RAM</b> 32 GB DDR5</span>
          <span class="chip"><b>MB</b> MSI MAG B650 Tomahawk</span>
        </div>
      </div>
      <div class="meta-right">
        <div class="meta-date">{date}</div>
        <div class="meta-sw">llama.cpp <span>b9858</span></div>
        <div class="meta-sw">CUDA <span>13.3</span> &middot; Blackwell <span>sm_120a</span></div>
        <div class="meta-sw" style="margin-top:6px">{runcount}-run average per cell</div>
      </div>
    </div>
  </div>
</header>
<main>
  <div class="wrap">
    <div class="stats">
      <div class="stat">
        <div class="stat-label">Peak decode</div>
        <div class="stat-value">{peak_tps}</div>
        <div class="stat-sub">tok/s &mdash; {peak_model}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Best dual-GPU lift</div>
        <div class="stat-value">{best_lift}&times;</div>
        <div class="stat-sub">{lift_desc}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Models tested</div>
        <div class="stat-value dim">{n_models}</div>
        <div class="stat-sub">{model_sizes}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Configurations</div>
        <div class="stat-value dim">5</div>
        <div class="stat-sub">Ollama &middot; GPU0 &middot; GPU1 &middot; Dual &middot; Dual tensor</div>
      </div>
    </div>

    <div class="metric-bar" id="metric-bar">
      <span class="metric-bar-lbl">Metric</span>
    </div>
    <div class="metric-hint" id="metric-hint"></div>

    <div class="tabs" id="tabs"></div>
    <div class="table-prompt" id="tier-prompt"></div>
    <div class="table-wrap"><table id="results-table"></table></div>

    <div class="legend">
      <div class="legend-item">
        <div class="legend-swatch" style="background:#1e3028;border:1px solid #2a5040"></div>
        faster / higher
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background:#1a1308;border:1px solid var(--amber)"></div>
        <span style="color:var(--amber)">&otimes;</span> CPU RAM spill (VRAM overflow &rarr; DDR5 ~90 GB/s)
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background:#1a1020;border:1px solid #3a2855"></div>
        <span style="color:#a070d0">MoE</span> &mdash; only active experts read per token
      </div>
      <div class="legend-item" style="margin-left:auto">
        <span style="font-family:var(--mono);font-size:9px">&middot;N</span>&nbsp;
        run count &mdash; <span style="color:var(--amber)">amber</span> = fewer than {runcount} runs
      </div>
    </div>

    <div class="findings">
      <div class="section-title">Key observations</div>
      <ul class="finding-list">
        <li class="finding">
          <span class="finding-label">Model size</span>
          <span class="finding-body">Dense models above ~16 GB overflow a single 16 GB GPU into system RAM, dropping decode to <span class="num-amber">4&ndash;10 tok/s</span> (DDR5 ~90 GB/s vs GPU 672 GB/s). Dual GPU eliminates the spill: Gemma 31B goes from <span class="num-amber">4.4</span> to <span class="num">29.2 tok/s</span> &mdash; a <strong>6.6&times; lift</strong>. MoE models are an exception: despite large GGUF files, only ~10&ndash;15% of weights are read per token (active experts), so they reach <span class="num">60&ndash;67 tok/s</span> even on a single GPU.</span>
        </li>
        <li class="finding">
          <span class="finding-label">PCIe lanes</span>
          <span class="finding-body">GPU0 (PCIe 4.0 x16 from CPU, 32 GB/s) outperforms GPU1 (PCIe 4.0 x2 from chipset, 4 GB/s) by <strong>1&ndash;8%</strong> in decode for VRAM-resident models. For spilling models, both GPUs are <strong>identical</strong> despite the 8&times; PCIe bandwidth gap &mdash; the bottleneck is CPU-side computation on off-GPU layers, not PCIe transfer. TTFT is slower on GPU1 for large prompts due to chipset pipeline latency.</span>
        </li>
        <li class="finding">
          <span class="finding-label">Backend</span>
          <span class="finding-body">For small dense models, llama.cpp single GPU is <strong>2&ndash;5% faster</strong> in decode. For MoE on a single GPU, Ollama wins by <strong>25&ndash;45%</strong> (Qwen 35B MoE: <span class="num">94.5</span> vs <span class="num-amber">67.0 tok/s</span>) via cuBLAS tensor cores tuned for sparse routing. For 27B+ dense models, llama.cpp dual tensor surpasses Ollama by up to <strong>67%</strong>. Ollama cannot split one inference across two GPUs.</span>
        </li>
        <li class="finding">
          <span class="finding-label">Dual GPU</span>
          <span class="finding-body">Small models that fit in one GPU: dual provides <strong>no benefit</strong> (1&ndash;3% penalty). Large dense models: dual is <strong>transformative</strong> &mdash; 2&ndash;7&times; by keeping the model in VRAM. MoE: a moderate <strong>35&ndash;55% gain</strong>. The second GPU is mandatory for 27B+ dense models at usable speeds on this hardware.</span>
        </li>
        <li class="finding">
          <span class="finding-label">Tensor split</span>
          <span class="finding-body"><code style="font-family:var(--mono);font-size:12px;color:var(--accent)">--split-mode tensor</code> runs both GPUs in parallel per layer, doubling effective bandwidth (2 &times; 672 = 1 344 GB/s). Dense models: <strong>50&ndash;70% speedup</strong> over layer split. MoE: <strong>12&ndash;16%</strong> &mdash; sparse activation limits per-token reads so allreduce overhead is proportionally larger. The PCIe x2 slot is not a bottleneck for decode: allreduce data is only ~14 KB per layer per token.</span>
        </li>
      </ul>
    </div>
  </div>
</main>
<footer>
  <div class="wrap"><div class="footer-inner">
    <span>Ollama + llama.cpp b9858 &middot; {runcount}-run average &middot; 2026</span>
    <span><a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> &middot; <a href="https://ollama.com">Ollama</a></span>
  </div></div>
</footer>
<script>
const RUNCOUNT = {runcount};
const TIERS    = {tiers_json};
const GMAX     = {global_max_json};

const COLS = [
  {{id:'ollama', label:'Ollama',     sub:'auto'}},
  {{id:'gpu0',   label:'LC GPU0',   sub:'MSI \xb7 x8'}},
  {{id:'gpu1',   label:'LC GPU1',   sub:'ASUS \xb7 x2'}},
  {{id:'dual',   label:'LC Dual',   sub:'layer split'}},
  {{id:'tensor', label:'LC Dual ⊗', sub:'tensor split', best:true}},
];

const METRICS = [
  {{id:'decode',  label:'Decode',    unit:'tok/s', higher:true,  fmt:v=>v.toFixed(1),
    hint:'Tokens generated per second — primary perf metric'}},
  {{id:'prefill', label:'Prefill',   unit:'tok/s', higher:true,  fmt:v=>v.toFixed(0),
    hint:'Prompt tokens processed per second — determines TTFT (run-2 KV cache warm)'}},
  {{id:'ttft',    label:'TTFT',      unit:'s',     higher:false, fmt:v=>v.toFixed(3),
    hint:'Time to first token in seconds — lower is better'}},
  {{id:'bw',      label:'Bandwidth', unit:'GB/s',  higher:true,  fmt:v=>v.toFixed(1),
    hint:'Effective memory bandwidth used — GGUF size \xd7 decode tok/s'}},
  {{id:'watts',   label:'Power',     unit:'W avg', higher:false, fmt:v=>v.toFixed(1),
    hint:'Average GPU power draw (both GPUs total) during inference — lower = more efficient',
    nodata: {no_watts_data}}},
];

let activeMet  = 'decode';
let activeTier = Object.keys(TIERS)[0];

function lerp(a,b,t){{return a+(b-a)*Math.max(0,Math.min(1,t))}}

function cellColor(val, spill, metId){{
  if(val===null) return{{bg:'transparent',fg:'var(--muted)'}};
  const met=METRICS.find(m=>m.id===metId);
  const maxV=GMAX[metId]||1;
  let t=Math.max(0,Math.min(1,val/maxV));
  if(!met.higher) t=1-t;
  if(spill&&met.higher){{
    return{{
      bg:`rgb(${{Math.round(lerp(18,32,t))}},${{Math.round(lerp(14,22,t))}},${{Math.round(lerp(6,10,t))}})`,
      fg:`hsl(35,${{Math.round(lerp(25,88,t))}}%,${{Math.round(lerp(30,65,t))}}%)`,
    }};
  }}
  return{{
    bg:`rgb(${{Math.round(lerp(16,10,t))}},${{Math.round(lerp(18,42,t))}},${{Math.round(lerp(26,34,t))}})`,
    fg:`hsl(162,${{Math.round(lerp(5,92,t))}}%,${{Math.round(lerp(28,62,t))}}%)`,
  }};
}}

function renderTable(){{
  const tier=TIERS[activeTier];
  const met=METRICS.find(m=>m.id===activeMet);

  document.getElementById('tier-prompt').innerHTML='<b>Prompt:</b> '+tier.prompt;
  document.getElementById('metric-hint').innerHTML=
    '<b>'+met.label+' ('+met.unit+')</b> — '+met.hint;

  let html='<thead><tr><th class="col-model">Model</th>';
  COLS.forEach(c=>{{
    html+=`<th class="col-val${{c.best?' col-best':''}}">
      ${{c.label}}<span class="th-sub">${{c.sub}}</span></th>`;
  }});
  html+='</tr></thead><tbody>';

  tier.rows.forEach(row=>{{
    const vals=COLS.map(c=>{{
      const d=row[c.id]||{{}};
      const v=d[activeMet];
      return(v!==null&&v!==undefined)?v:null;
    }});
    const valid=vals.filter(v=>v!==null);
    const rowBest=valid.length
      ?(met.higher?Math.max(...valid):Math.min(...valid))
      :null;

    html+=`<tr><td class="col-model">
      <div class="model-name">${{row.model}}</div>
      <div class="model-badges">
        ${{row.gguf?`<span class="badge">${{row.gguf}} GB</span>`:''}}
        <span class="badge">${{row.quant}}</span>
        ${{row.moe?'<span class="badge moe">MoE</span>':''}}
      </div></td>`;

    COLS.forEach(c=>{{
      const d=row[c.id]||{{}};
      const val=(d[activeMet]!==null&&d[activeMet]!==undefined)?d[activeMet]:null;
      const spill=d.spill||false;
      const runs=d.runs||0;
      const{{bg,fg}}=cellColor(val,spill,activeMet);
      const isBest=val!==null&&val===rowBest&&valid.length>1;
      const spillMark=(spill&&met.higher)?'<span class="spill-mark" style="color:var(--amber)">⊗</span>':'';
      const dispVal=val===null
        ?'<span class="cell-dash">—</span>'
        :`<span>${{met.fmt(val)}}</span>${{spillMark}}`;
      const runCls=runs>0&&runs<RUNCOUNT?' stale':'';
      const runBadge=runs>0?`<div class="run-ct${{runCls}}">\xb7${{runs}}</div>`:'';
      html+=`<td class="col-val" style="background:${{bg}}">
        <div class="cell-inner${{isBest?' best':''}}" style="color:${{fg}}">${{dispVal}}</div>
        ${{runBadge}}</td>`;
    }});
    html+='</tr>';
  }});
  html+='</tbody>';
  document.getElementById('results-table').innerHTML=html;
}}

// ── Metric pills ──────────────────────────────────────────────────────────────
const metricBar=document.getElementById('metric-bar');
METRICS.forEach((met,i)=>{{
  const btn=document.createElement('button');
  btn.className='mpill'+(i===0?' active':'');
  btn.dataset.met=met.id;
  btn.disabled=met.nodata===true;
  btn.title=met.nodata?'No GPU power data recorded (requires pynvml)':'';
  btn.innerHTML=met.label+'<span class="mpill-unit"> '+met.unit+'</span>';
  btn.addEventListener('click',()=>{{
    metricBar.querySelectorAll('.mpill').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    activeMet=met.id;
    renderTable();
  }});
  metricBar.appendChild(btn);
}});

// ── Tier tabs ─────────────────────────────────────────────────────────────────
const tabsEl=document.getElementById('tabs');
Object.entries(TIERS).forEach(([key,tier],i)=>{{
  const btn=document.createElement('button');
  btn.className='tab'+(i===0?' active':'');
  btn.dataset.tier=key;
  btn.innerHTML=(tier.label||key)+'<span class="tier-meta">'+tier.desc+'</span>';
  btn.addEventListener('click',()=>{{
    tabsEl.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    btn.classList.add('active');
    activeTier=key;
    renderTable();
  }});
  tabsEl.appendChild(btn);
}});

renderTable();
</script>
</body>
</html>
'''

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    DOCS_DIR.mkdir(exist_ok=True)
    rows  = load_data()
    tiers = build_tiers(rows)

    if not tiers:
        sys.exit("No data found. Run benchmarks first: python run.py run-all")

    gmax = compute_global_max(tiers)
    peak_tps, peak_model, best_lift, lift_desc, n_models = compute_stats(tiers)

    # detect whether any watts data exists
    has_watts = gmax["watts"] > 1.0
    if not has_watts:
        gmax["watts"] = 1.0  # prevent div-by-zero in JS

    import datetime
    date = datetime.date.today().isoformat()

    html = HTML_TEMPLATE.format(
        date           = date,
        runcount       = RUNCOUNT,
        peak_tps       = f"{peak_tps:.1f}",
        peak_model     = peak_model,
        best_lift      = f"{best_lift:.1f}",
        lift_desc      = lift_desc,
        n_models       = n_models,
        model_sizes    = "9B · 12B · 27B · 31B · 35B · incl. 2 MoE",
        global_max_json= json.dumps(gmax, separators=(",", ":")),
        tiers_json     = json.dumps(tiers, separators=(",", ":")),
        no_watts_data  = "true" if not has_watts else "false",
    )

    out = DOCS_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"  -> {out}")
    print(f"  {n_models} models · {len(tiers)} tiers · peak {peak_tps:.1f} tok/s")
    if not has_watts:
        print("  note: no GPU power data found in CSVs (requires pynvml + new benchmark run)")

if __name__ == "__main__":
    main()
