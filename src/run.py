"""
Dual-GPU LLM benchmark CLI.

Usage:
  python run.py gpus                                    show GPU VRAM + PCIe link state
  python run.py models                                  list models, file status, architecture
  python run.py env                                     print the environment manifest
  python run.py register                                register all models with Ollama
  python run.py register <id>                           register one model
  python run.py bench <id>                              run full benchmark matrix
  python run.py bench <id> --backend llamacpp
  python run.py bench <id> --gpu-configs dual dual_tensor --tiers chat
  python run.py bench <id> --repeats 10 --seed 7
  python run.py run-all
  python run.py run-all --models qwen3.5-9b-q4 gemma4-12b-qat --backend llamacpp
  python run.py results                                 legacy results table (see summarize.py)

Protocol (per model): each (backend, gpu_config) "unit" gets a fresh server; units
and tiers run in seeded-random order; each tier gets an untimed warm-up and then
--repeats timed requests. Every request carries a random nonce prefix and
cache_prompt=false so prefill is never served from cache, and ignore_eos so
every llama.cpp request generates exactly max_tokens.

Outputs per session (<ts>_<label>):
  results/runs/<session>.csv        one row per repetition (and per failure)
  results/runs/<session>.env.json   hardware/software manifest
  results/metrics/<session>.csv     per-cell mean + 95% CI (dashboard input)
  results/logs/<session>.log        human-readable console log
  results/responses/<session>.txt   model outputs (local only)
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import requests

from config import (GGUF_DIR, GPU_BW_PEAK_GBS, LLAMACPP_BIN, LLAMACPP_PORT, LOGS_DIR,
                    METRICS_DIR, MODELS, OLLAMA_API, OLLAMA_BIN, RESPONSES_DIR,
                    RESULTS_CSV, RESULTS_DIR, ROOT, RUNS_DIR)
from model_info import get_info, resolve_moe
from summarize import describe, write_metrics
from telemetry import (Sampler, environment, gpu_count, gpu_fields, gpu_temps,
                       vram_used_mib, wait_cool, write_environment)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for _d in (RESULTS_DIR, LOGS_DIR, METRICS_DIR, RUNS_DIR, RESPONSES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Protocol defaults ─────────────────────────────────────────────────────────
DEFAULT_REPEATS       = 5
DEFAULT_WARMUP        = 1
WARMUP_MAX_TOKENS     = 16      # warm-up exercises prefill + decode kernels, not full length
DEFAULT_COOLDOWN_S    = 10      # pause between units (server restarts)
DEFAULT_MAX_START_C   = 55      # wait for all GPUs to cool to this before each unit
REQUEST_TIMEOUT_S     = 1800    # spilling configs at ~4 tok/s × 1024 tokens need headroom
N_CTX                 = 8192

# ── Tee: write to stdout and a log file simultaneously ────────────────────────
class _Tee:
    def __init__(self, file):
        self._file   = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    # make sys.stdout happy
    @property
    def encoding(self):    return self._stdout.encoding
    @property
    def errors(self):      return self._stdout.errors

# ── Prompt tiers ──────────────────────────────────────────────────────────────
_dracula_path = ROOT / "dracula_ch1.txt"
if not _dracula_path.exists():
    sys.exit(f"Missing {_dracula_path}  — needed for prompt tiers.")

_words = _dracula_path.read_text(encoding="utf-8").split()

def _tier(n_words, question):
    return " ".join(_words[:n_words]) + "\n\n" + question

CODE_PROMPT = """\
Implement a Python class `RateLimiter` that enforces a sliding-window rate limit.

Requirements:
- `__init__(self, max_calls: int, period: float)` — allow at most `max_calls` calls in any `period`-second window
- `__call__(self, fn)` — decorator that wraps a function, raising `RateLimitExceeded` if the limit is hit
- `acquire(self)` — context manager that blocks (with `time.sleep`) until a slot is available
- Thread-safe using `threading.Lock`
- No external dependencies

Include a short usage example in a `if __name__ == "__main__"` block."""

BENCH_PROMPTS = {
    "chat": _tier(400,
        "Based on Jonathan Harker's journal entries above, "
        "summarize his first impressions of Eastern Europe "
        "and the cultural differences he observes."),
    "rag": _tier(1600,
        "Based on Jonathan Harker's journal entries above, "
        "list every warning or sign of danger Harker encounters "
        "before reaching Castle Dracula, and explain what each one suggests."),
    "longdoc": _tier(3200,
        "Based on Jonathan Harker's journal entries above, "
        "provide a detailed analysis of the journey to Transylvania. "

        "What atmosphere does Bram Stoker create, what specific events "
        "suggest supernatural danger, and how does Harker respond to these experiences?"),
    "code": CODE_PROMPT,
}

MAX_TOKENS_BY_TIER = {"chat": 512, "rag": 1024, "longdoc": 1024, "code": 1024}

def _nonced(prompt, rng):
    # fixed-width random prefix → no KV-prefix reuse across requests (M2), for
    # both backends; Ollama has no cache_prompt switch, so this is its only defence
    return f"[ref {rng.getrandbits(32):08x}]\n{prompt}"

# ── GPU configs ───────────────────────────────────────────────────────────────
GPU_CONFIGS = {
    #                    visible   ts     split-mode
    "single0":     ("0",   None,  None),
    "single1":     ("1",   None,  None),
    "dual":        ("0,1", "1,1", None),       # layer split 1:1
    "dual_tensor": ("0,1", "1,1", "tensor"),   # tensor parallelism 1:1
}
DEFAULT_GPU_CONFIGS = ("single0", "single1", "dual", "dual_tensor")

def get_model(id_):
    m = next((m for m in MODELS if m["id"] == id_), None)
    if not m:
        sys.exit(f"Unknown model '{id_}'. Available: {[m['id'] for m in MODELS]}")
    return m

# ── Session: log + per-rep runs CSV + manifest ────────────────────────────────
RUN_FIELDS = [
    "session", "started_at", "unit_order", "model_id", "model_name", "moe", "gguf_file",
    "backend", "gpu_config", "tier", "rep",
    "flash_attn", "mmap", "mtp_n", "ignore_eos", "cache_prompt", "max_tokens",
    "ok", "error", "error_phase",
    "load_time_s", "prompt_n", "cached_n", "n_generated",
    "decode_tok_s", "prefill_tok_s", "ttft_s", "wall_s",
    "energy_j", "decode_watts", "decode_j_per_tok", "avg_watts", "peak_watts", "power_gpus",
    "throttle", "start_temps", "peak_rss_mib", "observed_gpus", "ollama_vram_frac",
    "active_bytes", "bw_gb_s", "bw_pct",
]

class Session:
    def __init__(self, label, args):
        self.id        = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{label}"
        self.args      = args
        self.rows      = []
        self.fields    = RUN_FIELDS + gpu_fields(gpu_count())
        self.runs_path = RUNS_DIR / f"{self.id}.csv"
        self.log_path  = LOGS_DIR / f"{self.id}.log"
        self.resp_path = RESPONSES_DIR / f"{self.id}.txt"
        self._runs_fh = self._resp_fh = None

    def record(self, row):
        row = {"session": self.id, **row}
        self.rows.append(row)
        self._runs_w.writerow(row)
        self._runs_fh.flush()         # crash-safe: every rep hits disk immediately

    def save_response(self, model_id, backend, gpu_config, tier, rep, prompt, text):
        f = self._resp_fh
        f.write(f"{'━'*60}\nmodel:   {model_id}\nbackend: {backend}\ngpu:     {gpu_config}\n"
                f"tier:    {tier}\nrun:     {rep}\nts:      {time.strftime('%Y-%m-%d_%H-%M-%S')}\n")
        f.write("\n─── prompt ──────────────────────────────────────────────\n")
        f.write(prompt)
        f.write("\n─── response ────────────────────────────────────────────\n")
        f.write(text or "")
        f.write("\n\n")
        f.flush()

@contextmanager
def _session(label, args):
    s = Session(label, args)
    env = write_environment(RUNS_DIR / f"{s.id}.env.json", args)
    with open(s.log_path, "w", encoding="utf-8") as lf, \
         open(s.resp_path, "w", encoding="utf-8") as rf, \
         open(s.runs_path, "w", encoding="utf-8", newline="") as cf:
        lf.write(f"# schema: v2\n# {s.id}\n\n")      # log2csv skips v2 logs
        rf.write(f"# {s.id}\n\n")
        s._resp_fh, s._runs_fh = rf, cf
        s._runs_w = csv.DictWriter(cf, fieldnames=s.fields, extrasaction="ignore")
        s._runs_w.writeheader()
        old_stdout, sys.stdout = sys.stdout, _Tee(lf)
        try:
            _print_env_banner(env)
            yield s
        finally:
            _print_failures(s.rows)
            path = write_metrics(s.id, s.rows)
            print(f"\n  runs    → {s.runs_path}")
            print(f"  metrics → {path}")
            print(f"  env     → {RUNS_DIR / (s.id + '.env.json')}")
            print(f"  log     → {s.log_path}")
            sys.stdout = old_stdout

def _print_env_banner(env):
    git = env["harness_git"]
    print(f"  harness {git['version']} ({git['commit'][:10]}){' (DIRTY — uncommitted changes)' if git['dirty'] else ''}"
          f"  |  driver {env['nvidia_driver']}  |  {(env['llamacpp_version'] or 'llama.cpp n/a').splitlines()[0]}")
    for g in env["gpus"]:
        print(f"  GPU{g['index']} {g['name']}  {g['pci_bus_id']}  "
              f"PCIe max gen{g['pcie_gen_max']} x{g['pcie_width_max']}  "
              f"limit {g['power_limit_w'] or 0:.0f}W  power {'ok' if g['power_readable'] else 'N/A'}")

def _print_failures(rows):
    fails = [r for r in rows if r.get("ok") is False]
    if not fails:
        return
    print(f"\n  ⚠ {len(fails)} failed repetitions:")
    seen = set()
    for r in fails:
        k = (r["model_id"], r["backend"], r["gpu_config"], r["tier"], r.get("error_phase"))
        if k in seen:
            continue
        seen.add(k)
        print(f"    {r['model_id']} {r['backend']}/{r['gpu_config']} {r['tier']} "
              f"[{r.get('error_phase')}]: {(r.get('error') or '').splitlines()[0][:140]}")

# ── Legacy master CSV (analyze.py / `run.py results`) ─────────────────────────
RESULT_FIELDS = [
    "id", "ollama_tag", "moe", "backend", "gpu_config", "prompt_tier", "mtp_n",
    "flash_attn", "mmap",
    "ok", "error", "load_time_s",
    "decode_tok_per_s", "prompt_tok_per_s", "ttft_s",
    "n_generated", "peak_vram_mib",
    "peak_watts", "avg_watts",
    "peak_ram_mib", "ram_total_mib",
    "gguf_size_gb", "bandwidth_gb_s", "bandwidth_pct",
]

def save_result(row):
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

# ── Bandwidth (M7): bytes actually read per token, not GGUF size ──────────────
def _effective_bw_peak(gpu_config, observed_gpus):
    # Tensor split: both GPUs read their half simultaneously → n× bandwidth.
    # Layer split / single: GPUs work sequentially → one GPU's bandwidth.
    _, _, split_mode = GPU_CONFIGS.get(gpu_config, (None, None, None))
    n = len([g for g in observed_gpus.split(",") if g]) if observed_gpus else 1
    return GPU_BW_PEAK_GBS * (max(n, 1) if split_mode == "tensor" else 1)

def _bandwidth(active_bytes, decode_tok_s, gpu_config, observed_gpus):
    if not active_bytes or not decode_tok_s:
        return None, None
    bw = active_bytes / 1e9 * decode_tok_s
    return bw, bw / _effective_bw_peak(gpu_config, observed_gpus) * 100

# ── Ollama helpers ────────────────────────────────────────────────────────────
def _ollama_up():
    if not OLLAMA_BIN:
        return False
    try:
        requests.get(f"{OLLAMA_API}/api/version", timeout=3)
        return True
    except requests.RequestException:
        return False

def _ollama_registered(tag):
    """True/False, or None when Ollama isn't installed/running."""
    if not _ollama_up():
        return None
    r = requests.post(f"{OLLAMA_API}/api/show", json={"name": tag}, timeout=5)
    return r.status_code == 200

def _ollama_unload_all():
    """Evict every Ollama-resident model so it holds no VRAM during llama.cpp runs."""
    if not _ollama_up():
        return
    try:
        loaded = [m["name"] for m in requests.get(f"{OLLAMA_API}/api/ps", timeout=3).json().get("models", [])]
    except Exception:
        return
    for name in loaded:
        subprocess.run([OLLAMA_BIN, "stop", name], capture_output=True, timeout=30)
    for _ in range(30):
        try:
            if not requests.get(f"{OLLAMA_API}/api/ps", timeout=3).json().get("models"):
                return
        except Exception:
            return
        time.sleep(0.5)

def _ollama_vram_frac(tag):
    try:
        for m in requests.get(f"{OLLAMA_API}/api/ps", timeout=3).json().get("models", []):
            if m["name"] == tag and m.get("size"):
                return m.get("size_vram", 0) / m["size"]
    except Exception:
        pass
    return None

def _ollama_request(tag, prompt, max_tokens):
    payload = {"model": tag, "prompt": prompt, "stream": False, "keep_alive": "30m",
               "options": {"num_predict": max_tokens, "temperature": 0, "seed": 0}}
    r = requests.post(f"{OLLAMA_API}/api/generate", json=payload, timeout=REQUEST_TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(f"ollama HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    dec_s = d["eval_duration"] / 1e9
    pre_s = d["prompt_eval_duration"] / 1e9
    return {"decode_tok_s":  d["eval_count"] / dec_s if dec_s else None,
            "prefill_tok_s": d["prompt_eval_count"] / pre_s if pre_s else None,
            "ttft_s":        pre_s,
            "decode_s":      dec_s,
            "prompt_n":      d["prompt_eval_count"],
            "cached_n":      None,
            "n_generated":   d["eval_count"],
            "load_time_s":   d.get("load_duration", 0) / 1e9,
            "text":          d.get("response", "")}

# ── llama-server ──────────────────────────────────────────────────────────────
@contextmanager
def llama_server(gguf_path, gpu_config, mtp_n=0, mtp_pmin=0.75, flash_attn="on", mmap="on"):
    if not LLAMACPP_BIN.exists():
        raise FileNotFoundError(f"llama-server not found at {LLAMACPP_BIN} — set LLAMACPP_BIN in config.toml")

    visible, tensor_split, split_mode = GPU_CONFIGS[gpu_config]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    cmd = [str(LLAMACPP_BIN), "-m", str(gguf_path), "-ngl", "-1",
           "--port", str(LLAMACPP_PORT), "--host", "127.0.0.1",
           "-c", str(N_CTX), "--flash-attn", flash_attn, "-ub", "2048", "--log-disable"]
    if mmap == "off":
        cmd += ["--no-mmap"]
    if tensor_split:
        cmd += ["--tensor-split", tensor_split]
    if split_mode:
        cmd += ["--split-mode", split_mode]
    if mtp_n > 0:
        cmd += ["--spec-type", "draft-mtp",
                "--spec-draft-n-max", str(mtp_n),
                "--spec-draft-p-min", str(mtp_pmin)]
        # NOTE: --spec-type draft-mtp requires model trained with MTP heads.
        # Standard GGUFs (non-UD) won't have them → server crashes with
        # "context type MTP requested but model doesn't contain MTP layers"

    stderr_file = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8")
    log_path = Path(stderr_file.name)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=stderr_file)

    try:
        health = f"http://127.0.0.1:{LLAMACPP_PORT}/health"
        started = False
        for tick in range(720):
            time.sleep(0.5)
            if proc.poll() is not None:
                stderr_file.flush()
                err = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise RuntimeError(f"llama-server crashed (exit {proc.returncode}).\n{err}")
            if tick % 20 == 19:
                print(f"  {(tick+1)*0.5:.0f}s ...", end=" ", flush=True)
            try:
                if requests.get(health, timeout=2).status_code == 200:
                    started = True
                    break
            except (requests.ConnectionError, requests.exceptions.ReadTimeout):
                pass
        if not started:
            raise TimeoutError("llama-server did not become ready within 360s")
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        stderr_file.close()
        try:
            log_path.unlink()
        except Exception:
            pass

def _llamacpp_request(prompt, max_tokens, ignore_eos):
    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "seed": 0, "stream": False,
               "cache_prompt": False, "ignore_eos": ignore_eos}
    r = requests.post(f"http://127.0.0.1:{LLAMACPP_PORT}/v1/chat/completions",
                      json=payload, timeout=REQUEST_TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(f"llama-server HTTP {r.status_code}: {r.text[:300]}")
    d = r.json(); t = d.get("timings", {})
    msg      = d.get("choices", [{}])[0].get("message", {})
    thinking = msg.get("reasoning_content", "")
    content  = msg.get("content", "") or ""
    return {"decode_tok_s":  t.get("predicted_per_second"),
            "prefill_tok_s": t.get("prompt_per_second"),
            "ttft_s":        t.get("prompt_ms", 0) / 1000,
            "decode_s":      t.get("predicted_ms", 0) / 1000,
            "prompt_n":      t.get("prompt_n"),
            "cached_n":      t.get("cache_n"),
            "n_generated":   t.get("predicted_n") or d.get("usage", {}).get("completion_tokens"),
            "text":          (f"<think>\n{thinking}\n</think>\n\n{content}".strip()
                              if thinking else content)}

# ── One timed repetition ──────────────────────────────────────────────────────
def _timed_rep(ctx, rep, request_fn, pid=None, proc_name=None):
    """Run one request under telemetry; returns the runs-CSV row (ok or failed)."""
    base = {**ctx["row"], "rep": rep, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "start_temps": ";".join(f"{i}:{t}" for i, t in gpu_temps().items())}
    prompt = _nonced(ctx["prompt"], ctx["rng"])
    smp = Sampler(pid=pid, proc_name=proc_name).start()
    try:
        res = request_fn(prompt)
    except Exception as e:
        smp.stop()
        return {**base, "ok": False, "error": f"{e.__class__.__name__}: {e}", "error_phase": "request"}
    smp.stop()

    tel = smp.summary(ctx["baseline_vram"])
    dec_w = smp.tail_watts(res.get("decode_s"))
    bw, bw_pct = _bandwidth(ctx["active_bytes"], res["decode_tok_s"],
                            ctx["row"]["gpu_config"], tel["observed_gpus"])
    ctx["session"].save_response(ctx["row"]["model_id"], ctx["row"]["backend"],
                                 ctx["row"]["gpu_config"], ctx["row"]["tier"], rep, prompt, res["text"])
    return {**base, **tel, "ok": True,
            **{k: res.get(k) for k in ("decode_tok_s", "prefill_tok_s", "ttft_s",
                                       "prompt_n", "cached_n", "n_generated")},
            "load_time_s":      ctx.get("load_time_s"),
            "decode_watts":     dec_w,
            "decode_j_per_tok": (dec_w / res["decode_tok_s"]) if dec_w and res["decode_tok_s"] else None,
            "bw_gb_s": bw, "bw_pct": bw_pct}

def _run_tier(ctx, tier, request_fn, opts, pid=None, proc_name=None):
    """Warm-up + timed repetitions for one tier of one unit. Returns ok rows."""
    s, row = ctx["session"], ctx["row"]
    max_tokens = MAX_TOKENS_BY_TIER[tier]
    ctx = {**ctx, "prompt": BENCH_PROMPTS[tier],
           "row": {**row, "tier": tier, "max_tokens": max_tokens}}
    print(f"\n    ── {tier} ──")
    for w in range(opts.warmup):
        try:
            request_fn(_nonced(BENCH_PROMPTS[tier], ctx["rng"]), WARMUP_MAX_TOKENS)
        except Exception as e:
            print(f"    warm-up failed: {e}")
    ok_rows = []
    for rep in range(1, opts.repeats + 1):
        print(f"    run {rep}/{opts.repeats} ...", end=" ", flush=True)
        r = _timed_rep(ctx, rep, lambda p: request_fn(p, max_tokens), pid=pid, proc_name=proc_name)
        s.record(r)
        if r["ok"]:
            ok_rows.append(r)
            cache_note = f"  cached {r['cached_n']}!" if r.get("cached_n") else ""
            print(f"{r['decode_tok_s']:.1f} tok/s  (prompt {r['prefill_tok_s']:.0f} tok/s  "
                  f"TTFT {r['ttft_s']:.2f}s  n={r['n_generated']}){cache_note}")
        else:
            print(f"FAILED: {r['error'].splitlines()[0][:140]}")
            if pid is not None and ctx.get("proc") and ctx["proc"].poll() is not None:
                print("    server died — skipping remaining reps")
                for rest in range(rep + 1, opts.repeats + 1):
                    s.record({**ctx["row"], "rep": rest, "ok": False,
                              "error": "server died earlier in unit", "error_phase": "request"})
                break
    _print_cell(ok_rows)
    _save_legacy(ctx["row"], ok_rows, ctx)
    return ok_rows

def _print_cell(rows):
    if not rows:
        return
    d = describe([r["decode_tok_s"] for r in rows])
    p = describe([r["prefill_tok_s"] for r in rows])
    t = describe([r["ttft_s"] for r in rows])
    ci = lambda x: f" ± {x['ci95']:.1f}" if x["ci95"] is not None else ""
    print(f"    decode  {d['mean']:>7.1f}{ci(d)} tok/s   (CV {d['cv']*100 if d['cv'] else 0:.1f}%, n={d['n']})")
    print(f"    prompt  {p['mean']:>7.0f}{ci(p)} tok/s  TTFT {t['mean']:.3f}s")
    last = rows[-1]
    gpus = [k for k in last if k.endswith("_vram_mib") and not k.endswith("delta_mib")]
    vram = "  ".join(f"GPU{k[3:k.index('_')]} {last[k]/1024:.1f}G" for k in sorted(gpus) if last[k])
    print(f"    vram    {vram}   on GPUs [{last['observed_gpus']}]"
          + (f"   ollama in-VRAM {last['ollama_vram_frac']*100:.0f}%" if last.get("ollama_vram_frac") is not None else ""))
    if last.get("decode_j_per_tok"):
        e = describe([r["decode_j_per_tok"] for r in rows if r.get("decode_j_per_tok")])
        print(f"    energy  {e['mean']:.2f} J/tok decode  ({last['decode_watts']:.0f} W, GPUs {last['power_gpus']})")
    if any(r.get("throttle") for r in rows):
        print(f"    ⚠ throttled: {sorted({r['throttle'] for r in rows if r.get('throttle')})}")

def _save_legacy(row, ok_rows, ctx):
    mean = lambda k: describe([r.get(k) for r in ok_rows])["mean"] if ok_rows else None
    vram = {int(k[3:k.index("_")]): mean(k) for k in (ok_rows[-1] if ok_rows else {})
            if k.startswith("gpu") and k.endswith("_vram_mib")}
    save_result({"id": row["model_id"], "ollama_tag": ctx["model"]["ollama_tag"], "moe": row["moe"],
                 "backend": row["backend"], "gpu_config": row["gpu_config"], "prompt_tier": row["tier"],
                 "mtp_n": row["mtp_n"], "flash_attn": row["flash_attn"], "mmap": row["mmap"],
                 "ok": bool(ok_rows), "error": "" if ok_rows else "all repetitions failed",
                 "load_time_s": ctx.get("load_time_s"),
                 "decode_tok_per_s": mean("decode_tok_s"), "prompt_tok_per_s": mean("prefill_tok_s"),
                 "ttft_s": mean("ttft_s"), "n_generated": mean("n_generated"),
                 "peak_vram_mib": vram or None, "peak_watts": mean("peak_watts"),
                 "avg_watts": mean("avg_watts"), "peak_ram_mib": mean("peak_rss_mib"),
                 "gguf_size_gb": ctx["gguf_gb"], "bandwidth_gb_s": mean("bw_gb_s"),
                 "bandwidth_pct": mean("bw_pct")})

# ── Units ─────────────────────────────────────────────────────────────────────
def _pre_unit(opts):
    if opts.cooldown:
        time.sleep(opts.cooldown)
    temps = wait_cool(opts.max_start_temp)
    return temps

def _unit_llamacpp(ctx, cfg, tiers, opts):
    s = ctx["session"]
    _ollama_unload_all()
    _pre_unit(opts)
    ctx = {**ctx, "baseline_vram": vram_used_mib(),
           "row": {**ctx["row"], "backend": "llamacpp", "gpu_config": cfg}}
    mtp_tag = f" mtp={opts.mtp_n}" if opts.mtp_n > 0 else ""
    print(f"\n  [llamacpp | cfg={cfg} fa={opts.flash_attn} mmap={opts.mmap}{mtp_tag}]")
    print(f"    starting llama-server ...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        with llama_server(ctx["model"]["gguf"], cfg, mtp_n=opts.mtp_n,
                          flash_attn=opts.flash_attn, mmap=opts.mmap) as proc:
            ctx |= {"load_time_s": time.perf_counter() - t0, "proc": proc}
            print(f"ready in {ctx['load_time_s']:.1f}s")
            req = lambda p, n: _llamacpp_request(p, n, opts.ignore_eos == "on")
            for tier in tiers:
                _run_tier(ctx, tier, req, opts, pid=proc.pid)
    except (TimeoutError, RuntimeError, FileNotFoundError) as e:
        msg = f"{e.__class__.__name__}: {e}"
        print(f"\n    → startup failed: {msg.splitlines()[0][:200]}")
        for tl in (msg.splitlines()[1:] if "crashed" in msg else []):
            if "assert" in tl.lower() or "error" in tl.lower():
                print(f"      {tl.strip()[:200]}")
        for tier in tiers:
            s.record({**ctx["row"], "tier": tier, "max_tokens": MAX_TOKENS_BY_TIER[tier],
                      "rep": 0, "ok": False, "error": msg[-1500:], "error_phase": "startup"})

def _unit_ollama(ctx, tiers, opts):
    m, s = ctx["model"], ctx["session"]
    reg = _ollama_registered(m["ollama_tag"])
    if reg is None:
        print("\n  [ollama] not installed/running — skipped")
        return
    if not reg:
        print(f"\n  [ollama] not registered — run: python run.py register {m['id']}")
        return
    _ollama_unload_all()
    _pre_unit(opts)
    ctx = {**ctx, "baseline_vram": vram_used_mib(),
           "row": {**ctx["row"], "backend": "ollama", "gpu_config": "auto",
                   "flash_attn": "", "mmap": "", "mtp_n": 0, "ignore_eos": "off", "cache_prompt": "nonce"}}
    print(f"\n  [ollama]")
    t0 = time.perf_counter()
    try:
        load = _ollama_request(m["ollama_tag"], "hi", 1)
        ctx["load_time_s"] = load["load_time_s"] or (time.perf_counter() - t0)
        print(f"    loaded in {ctx['load_time_s']:.1f}s")
        frac = _ollama_vram_frac(m["ollama_tag"])
        ctx["row"]["ollama_vram_frac"] = frac
        req = lambda p, n: _ollama_request(m["ollama_tag"], p, n)
        for tier in tiers:
            _run_tier(ctx, tier, req, opts, proc_name="ollama")
    except Exception as e:
        msg = f"{e.__class__.__name__}: {e}"
        print(f"    → failed: {msg[:200]}")
        for tier in tiers:
            s.record({**ctx["row"], "tier": tier, "max_tokens": MAX_TOKENS_BY_TIER[tier],
                      "rep": 0, "ok": False, "error": msg, "error_phase": "startup"})
    finally:
        _ollama_unload_all()

def _bench_model(session, m, opts):
    info = get_info(m["gguf"])
    moe  = resolve_moe(m, info)
    rng  = random.Random(f"{opts.seed}:{m['id']}")
    _title = f"{m['id']}  —  {m['name']}"
    print(f"\n{'='*max(55,len(_title)+4)}\n  {_title}\n{'='*max(55,len(_title)+4)}")
    if info:
        print(f"  {info['architecture']}  layers={info['n_layers']}  hidden={info['hidden_dim']}  "
              f"moe={moe}  read/token={info['active_bytes']/1e9:.2f} GB  GGUF={info['total_bytes']/1e9:.2f} GB")
    elif not m["gguf"].exists():
        print(f"  gguf not found: {m['gguf']}")

    units = []
    if "ollama" in opts.backends:
        units.append(("ollama", "auto"))
    if "llamacpp" in opts.backends:
        if m["gguf"].exists():
            units += [("llamacpp", c) for c in opts.gpu_configs]
        else:
            print(f"  [llamacpp] skipped — gguf not found")
    if opts.shuffle:
        rng.shuffle(units)
    print(f"  unit order: {', '.join(f'{b}/{c}' for b, c in units)}")

    base_ctx = {"session": session, "model": m, "rng": rng,
                "active_bytes": info["active_bytes"] if info else None,
                "gguf_gb": m["gguf"].stat().st_size / 1e9 if m["gguf"].exists() else None,
                "row": {"model_id": m["id"], "model_name": m["name"], "moe": moe,
                        "gguf_file": m["gguf"].name, "flash_attn": opts.flash_attn, "mmap": opts.mmap,
                        "mtp_n": opts.mtp_n, "ignore_eos": opts.ignore_eos, "cache_prompt": "off",
                        "active_bytes": info["active_bytes"] if info else None}}
    for order, (backend, cfg) in enumerate(units, 1):
        tiers = list(opts.tiers)
        if opts.shuffle:
            rng.shuffle(tiers)
        ctx = {**base_ctx, "row": {**base_ctx["row"], "unit_order": order}}
        if backend == "ollama":
            _unit_ollama(ctx, tiers, opts)
        else:
            _unit_llamacpp(ctx, cfg, tiers, opts)

# ── Subcommands ───────────────────────────────────────────────────────────────
def cmd_gpus(_args):
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            used, total = mem.used / 1024**3, mem.total / 1024**3
            bar  = "█" * int(used / total * 20)
            link = (f"PCIe gen{pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h)}"
                    f" x{pynvml.nvmlDeviceGetCurrPcieLinkWidth(h)} now"
                    f" (max gen{pynvml.nvmlDeviceGetMaxPcieLinkGeneration(h)}"
                    f" x{pynvml.nvmlDeviceGetMaxPcieLinkWidth(h)})")
            print(f"[{i}] {name}  {link}")
            print(f"    {used:.1f} / {total:.1f} GiB  [{bar:<20}]")
        print("\n  note: idle GPUs downclock the link; check pcie_gen/width columns in runs CSVs for under-load values")
    except Exception as e:
        print(f"pynvml error: {e}")

def cmd_models(_args):
    print(f"\n  {'id':<22} {'gguf file':<40} {'size':>6} {'arch':<11} {'moe':<4} {'read/tok':>8}  {'ollama'}")
    print("  " + "─" * 104)
    for m in MODELS:
        exists = m["gguf"].exists()
        info   = get_info(m["gguf"]) if exists else None
        size   = f"{m['gguf'].stat().st_size/1e9:.1f}G" if exists else "—"
        reg    = {True: "✓", False: "—", None: "n/a"}[_ollama_registered(m["ollama_tag"])]
        mark   = "✓" if exists else "✗"
        arch   = info["architecture"] if info else "?"
        moe    = "yes" if resolve_moe(m, info) else "no"
        rpt    = f"{info['active_bytes']/1e9:.2f}G" if info else "—"
        print(f"  {m['id']:<22} {mark} {m['gguf'].name:<38} {size:>6} {arch:<11} {moe:<4} {rpt:>8}  {reg}")
    print(f"\n  registry: models.toml   GGUF_DIR: {GGUF_DIR}\n")

def cmd_env(args):
    print(json.dumps(environment(args), indent=2, default=str))

def cmd_register(args):
    if not _ollama_up():
        sys.exit("Ollama is not installed/running — start it with: ollama serve")
    targets = MODELS if not getattr(args, "id", None) else [get_model(args.id)]
    for m in targets:
        tag = m["ollama_tag"]
        if _ollama_registered(tag):
            print(f"  {m['id']}  already registered as {tag}")
            continue
        if not m["gguf"].exists():
            print(f"  {m['id']}  SKIPPED — gguf not found: {m['gguf']}")
            continue
        mf = GGUF_DIR / f"{m['id']}.Modelfile"
        mf.write_text(f"FROM {m['gguf']}\n", encoding="utf-8")
        r = subprocess.run([OLLAMA_BIN, "create", tag, "-f", str(mf)],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0:
            print(f"  {m['id']}  registered as {tag} ✓")
        else:
            print(f"  {m['id']}  FAILED:\n{r.stderr.strip()}")

def _opts(args):
    args.backends    = ["ollama", "llamacpp"] if args.backend == "both" else [args.backend]
    args.tiers       = args.tiers or list(BENCH_PROMPTS)
    args.gpu_configs = args.gpu_configs or list(DEFAULT_GPU_CONFIGS)
    args.shuffle     = not args.no_shuffle
    if args.seed is None:
        args.seed = int(time.time())
    return args

def cmd_bench(args):
    opts = _opts(args)
    m = get_model(opts.id)
    label = "_".join([m["id"]] + (opts.tiers if len(opts.tiers) < 3 else []) +
                     (["_".join(opts.gpu_configs)] if opts.backend != "ollama" else []))
    with _session(label, opts) as s:
        print(f"  seed {opts.seed}  repeats {opts.repeats}  warmup {opts.warmup}  ignore_eos {opts.ignore_eos}")
        _bench_model(s, m, opts)

def cmd_run_all(args):
    opts = _opts(args)
    models = [get_model(i) for i in opts.models] if opts.models else MODELS
    with _session("run-all", opts) as s:
        print(f"  seed {opts.seed}  repeats {opts.repeats}  warmup {opts.warmup}  ignore_eos {opts.ignore_eos}")
        for m in models:
            _bench_model(s, m, opts)

def cmd_results(_args):
    if not RESULTS_CSV.exists():
        print("No results yet. Run: python run.py run-all")
        return

    rows = list(csv.DictReader(open(RESULTS_CSV)))
    ok   = [r for r in rows if r.get("ok") == "True"]
    if not ok:
        print("No successful results yet.")
        return

    ok.sort(key=lambda r: (r.get("id",""), r.get("prompt_tier",""),
                           r.get("backend",""), r.get("gpu_config","")))

    has_mtp = any(r.get("mtp_n","0") not in ("0","") for r in ok)
    mtp_hdr = f" {'mtp':>4}" if has_mtp else ""
    has_fa  = any(r.get("flash_attn","on") not in ("", "on") for r in ok)
    fa_hdr  = f" {'fa':>3}" if has_fa else ""
    has_mm  = any(r.get("mmap","on") not in ("", "on") for r in ok)
    mm_hdr  = f" {'mmap':>4}" if has_mm else ""
    print(f"\n  {'id':<24} {'tier':<9} {'backend':<10} {'gpu':<9}{mtp_hdr}{fa_hdr}{mm_hdr}"
          f" {'decode t/s':>10} {'prompt t/s':>11} {'TTFT s':>7} {'bw%':>6}")
    print("  " + "─" * (92 + (5 if has_mtp else 0) + (4 if has_fa else 0) + (5 if has_mm else 0)))
    for r in ok:
        def f(k, fmt):
            try: return fmt.format(float(r[k]))
            except: return f"{'—':>8}"
        mtp_col = f" {r.get('mtp_n','0'):>4}" if has_mtp else ""
        fa_col  = f" {r.get('flash_attn','on'):>3}" if has_fa else ""
        mm_col  = f" {r.get('mmap','on'):>4}" if has_mm else ""
        print(f"  {r.get('id','?'):<24} {r.get('prompt_tier','?'):<9}"
              f" {r.get('backend','?'):<10} {r.get('gpu_config','?'):<9}{mtp_col}{fa_col}{mm_col}"
              f" {f('decode_tok_per_s','{:>10.1f}')}"
              f" {f('prompt_tok_per_s','{:>11.0f}')}"
              f" {f('ttft_s','{:>7.3f}')}"
              f" {f('bandwidth_pct','{:>5.1f}%')}")

    print(f"\n  {len(ok)} results  |  with confidence intervals: python summarize.py")

# ── Argument parser ───────────────────────────────────────────────────────────
def _add_bench_args(p):
    p.add_argument("--backend",     choices=["ollama", "llamacpp", "both"], default="both")
    p.add_argument("--tiers",       nargs="+", choices=list(BENCH_PROMPTS),
                   help="Prompt tiers to run (default: all)")
    p.add_argument("--gpu-configs", nargs="+", choices=list(GPU_CONFIGS), dest="gpu_configs",
                   help=f"llama.cpp GPU configs (default: {' '.join(DEFAULT_GPU_CONFIGS)})")
    p.add_argument("--repeats",     type=int, default=DEFAULT_REPEATS,
                   help=f"timed repetitions per cell (default: {DEFAULT_REPEATS})")
    p.add_argument("--warmup",      type=int, default=DEFAULT_WARMUP,
                   help=f"untimed warm-up requests per tier (default: {DEFAULT_WARMUP})")
    p.add_argument("--seed",        type=int, default=None,
                   help="seed for unit/tier order and prompt nonces (default: time-based, recorded)")
    p.add_argument("--no-shuffle",  action="store_true", dest="no_shuffle",
                   help="run units and tiers in fixed order")
    p.add_argument("--cooldown",    type=float, default=DEFAULT_COOLDOWN_S,
                   help=f"seconds to pause before each unit (default: {DEFAULT_COOLDOWN_S})")
    p.add_argument("--max-start-temp", type=float, default=DEFAULT_MAX_START_C, dest="max_start_temp",
                   help=f"wait until all GPUs ≤ this °C before each unit (default: {DEFAULT_MAX_START_C})")
    p.add_argument("--ignore-eos",  choices=["on", "off"], default="on", dest="ignore_eos",
                   help="llama.cpp: always generate exactly max_tokens (default: on)")
    p.add_argument("--mtp",         type=int, default=0, dest="mtp_n", metavar="N",
                   help="MTP draft tokens (0=disabled; try 3–5 for ~2× decode speed)")
    p.add_argument("--flash-attn",  choices=["on", "off"], default="on", dest="flash_attn",
                   help="llama.cpp flash attention (default: on)")
    p.add_argument("--mmap",        choices=["on", "off"], default="on",
                   help="llama.cpp mmap for model weights (default: on)")

def main():
    p   = argparse.ArgumentParser(description="Dual-GPU LLM benchmark",
                                  formatter_class=argparse.RawDescriptionHelpFormatter,
                                  epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gpus",    help="Show GPU VRAM and PCIe link state")
    sub.add_parser("models",  help="List models, file status and architecture")
    sub.add_parser("env",     help="Print the environment manifest")
    sub.add_parser("results", help="Print legacy results table")

    pr = sub.add_parser("register", help="Register model(s) with Ollama")
    pr.add_argument("id", nargs="?", help="Model id (omit for all)")

    pb = sub.add_parser("bench", help="Benchmark one model")
    pb.add_argument("id", help="Model id  (python run.py models to list)")
    _add_bench_args(pb)

    pa = sub.add_parser("run-all", help="Benchmark all models (or --models subset)")
    pa.add_argument("--models", nargs="+", help="Model ids to include (default: all in models.toml)")
    _add_bench_args(pa)

    args = p.parse_args()
    {"gpus": cmd_gpus, "models": cmd_models, "env": cmd_env, "register": cmd_register,
     "bench": cmd_bench, "run-all": cmd_run_all, "results": cmd_results}[args.cmd](args)

if __name__ == "__main__":
    main()
