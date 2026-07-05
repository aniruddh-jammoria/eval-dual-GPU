"""
Dual-GPU LLM benchmark CLI.

Usage:
  python run.py gpus                                    show GPU VRAM state
  python run.py models                                  list models + file status
  python run.py register                                register all models with Ollama
  python run.py register <id>                           register one model
  python run.py bench <id>                              run full benchmark matrix
  python run.py bench <id> --backend llamacpp
  python run.py bench <id> --backend llamacpp --gpu dual --tiers chat
  python run.py bench <id> --gpu-configs dual dual_tensor --tiers chat
  python run.py run-all
  python run.py run-all --backend ollama --tiers chat rag
  python run.py results                                 print results table
"""

import argparse
import ast
import csv
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
GGUF_DIR     = Path(r"D:\llama-models")
LLAMACPP_BIN = Path(r"D:\llama.cpp\llama-server.exe")
LLAMACPP_PORT = 8080
OLLAMA_API   = "http://localhost:11434"
ROOT          = Path(__file__).resolve().parent.parent   # repo root (src/ lives under it)
RESULTS_DIR   = ROOT / "results"
RESULTS_CSV   = RESULTS_DIR / "results.csv"
LOGS_DIR      = RESULTS_DIR / "logs"
METRICS_DIR   = RESULTS_DIR / "metrics"
RESPONSES_DIR = RESULTS_DIR / "responses"
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

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

_response_file = None   # session-scoped response log, set by _log_session

@contextmanager
def _log_session(label):
    global _response_file
    ts           = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path     = LOGS_DIR     / f"{ts}_{label}.log"
    resp_path    = RESPONSES_DIR / f"{ts}_{label}.txt"
    with open(log_path, "w", encoding="utf-8") as f, \
         open(resp_path, "w", encoding="utf-8") as rf:
        rf.write(f"# {label}  started {ts}\n\n")
        _response_file = rf
        f.write(f"# {label}  started {ts}\n\n")
        tee = _Tee(f)
        old_stdout, sys.stdout = sys.stdout, tee
        try:
            yield log_path
        finally:
            sys.stdout = old_stdout
            _response_file = None
    # dump per-run CSV to results/metrics/ after the log is closed
    try:
        from log2csv import parse_log
        import csv as _csv
        rows = parse_log(log_path)
        if rows:
            csv_path = METRICS_DIR / (log_path.stem + ".csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                w = _csv.DictWriter(cf, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f"  metrics → {csv_path}")
    except Exception as e:
        print(f"  [log2csv] skipped: {e}")
    print(f"  log → {log_path}")

REPEATS          = 2
N_CTX            = 8192
GPU_BW_PEAK_GBS  = 672.0

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

# ── GPU configs ───────────────────────────────────────────────────────────────
GPU_CONFIGS = {
    #                    visible   ts     split-mode
    "single0":     ("0",   None,  None),
    "single1":     ("1",   None,  None),
    "dual":        ("0,1", "1,1", None),       # layer split 1:1
    "dual_tensor": ("0,1", "1,1", "tensor"),   # tensor parallelism 1:1
}
DEFAULT_GPU_CONFIGS = ("single0", "single1", "dual", "dual_tensor")

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = [
    {"id": "gemma4-12b-qat",     "name": "Gemma 4 12B IT QAT (Q4_0)",
     "ollama_tag": "eval/gemma4-12b-qat",
     "gguf": GGUF_DIR / "gemma-4-12b-it-qat-q4_0.gguf"},
    {"id": "qwen3.5-9b-q4",      "name": "Qwen3.5 9B (Q4_K_M)",
     "ollama_tag": "eval/qwen3.5-9b-q4",
     "gguf": GGUF_DIR / "Qwen3.5-9B-Q4_K_M.gguf"},
    {"id": "qwen3.5-9b-q8",      "name": "Qwen3.5 9B (Q8_0)",
     "ollama_tag": "eval/qwen3.5-9b-q8",
     "gguf": GGUF_DIR / "Qwen3.5-9B-Q8_0.gguf"},
    {"id": "qwen3.6-27b-q4",     "name": "Qwen3.6 27B (Q4_K_M)",
     "ollama_tag": "eval/qwen3.6-27b-q4",
     "gguf": GGUF_DIR / "Qwen3.6-27B-Q4_K_M.gguf"},
    {"id": "qwen3.6-27b-q6",     "name": "Qwen3.6 27B (Q6_K)",
     "ollama_tag": "eval/qwen3.6-27b-q6",
     "gguf": GGUF_DIR / "Qwen3.6-27B-Q6_K.gguf"},
    {"id": "gemma4-26b-moe-q4",  "name": "Gemma 4 26B A4B IT MoE UD (Q4_K_M)",
     "ollama_tag": "eval/gemma4-26b-moe-q4",
     "gguf": GGUF_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"},
    {"id": "qwen3.6-35b-a3b-q4", "name": "Qwen3.6 35B A3B MoE UD (Q4_K_M)",
     "ollama_tag": "eval/qwen3.6-35b-a3b-q4",
     "gguf": GGUF_DIR / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"},
    {"id": "gemma4-31b-q5",      "name": "Gemma 4 31B IT UD (Q5_K_XL)",
     "ollama_tag": "eval/gemma4-31b-q5",
     "gguf": GGUF_DIR / "gemma-4-31B-it-UD-Q5_K_XL.gguf"},
]

def get_model(id_):
    m = next((m for m in MODELS if m["id"] == id_), None)
    if not m:
        sys.exit(f"Unknown model '{id_}'. Available: {[m['id'] for m in MODELS]}")
    return m

# ── CSV schema ────────────────────────────────────────────────────────────────
RESULT_FIELDS = [
    "id", "ollama_tag", "backend", "gpu_config", "prompt_tier", "mtp_n",
    "ok", "error", "load_time_s",
    "decode_tok_per_s", "prompt_tok_per_s", "ttft_s",
    "n_generated", "peak_vram_mib",
    "peak_watts", "avg_watts",
    "gguf_size_gb", "bandwidth_gb_s", "bandwidth_pct",
]

def save_response(model_id, backend, gpu_config, tier, run_i, prompt, response_text):
    f = _response_file
    if f is None:
        return
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    f.write(f"{'━'*60}\n")
    f.write(f"model:   {model_id}\n")
    f.write(f"backend: {backend}\n")
    f.write(f"gpu:     {gpu_config}\n")
    f.write(f"tier:    {tier}\n")
    f.write(f"run:     {run_i}\n")
    f.write(f"ts:      {ts}\n")
    f.write("\n─── prompt ──────────────────────────────────────────────\n")
    f.write(prompt)
    f.write("\n─── response ────────────────────────────────────────────\n")
    f.write(response_text)
    f.write("\n\n")
    f.flush()

def save_result(row):
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

# ── VRAM sampler ──────────────────────────────────────────────────────────────
class VramSampler:
    def __init__(self):
        self.peak = {}
        self._stop = threading.Event()

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i)
                       for i in range(pynvml.nvmlDeviceGetCount())}
            while not self._stop.is_set():
                for i, h in handles.items():
                    mib = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
                    self.peak[i] = max(self.peak.get(i, 0), mib)
                self._stop.wait(0.25)
        except Exception:
            pass

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

# ── Power sampler ─────────────────────────────────────────────────────────────
# NOTE: some consumer cards don't expose power telemetry to NVML (returns
# NVMLError_NotSupported — confirmed on the MSI VENTUS RTX 5060 Ti / GPU0 here,
# while the ASUS DUAL / GPU1 reports fine). We sample each GPU independently and
# skip the ones that fail, so a single unsupported GPU no longer zeroes out the
# whole measurement. `supported` records which physical indices actually report.
class PowerSampler:
    def __init__(self):
        self.peak = 0.0            # peak total watts (sum of supported GPUs)
        self.avg  = 0.0            # mean total watts
        self.per_gpu_peak = {}     # physical index → peak watts
        self.supported = set()     # physical indices that report power
        self.n_gpus = 0
        self._samples = []
        self._stop = threading.Event()

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i)
                       for i in range(pynvml.nvmlDeviceGetCount())}
            self.n_gpus = len(handles)
            while not self._stop.is_set():
                total = 0.0
                for i, h in handles.items():
                    try:
                        w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000   # mW → W
                    except pynvml.NVMLError:
                        continue   # this GPU doesn't expose power — skip it
                    total += w
                    self.supported.add(i)
                    self.per_gpu_peak[i] = max(self.per_gpu_peak.get(i, 0.0), w)
                self._samples.append(total)
                self.peak = max(self.peak, total)
                self._stop.wait(0.5)
        except Exception:
            pass

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()
        self.avg = sum(self._samples) / len(self._samples) if self._samples else 0.0

def _effective_bw_peak(gpu_config, peak_vram_mib):
    # Tensor split: both GPUs read their half simultaneously → 2× bandwidth
    # Layer split / single: sequential → peak is one GPU's bandwidth regardless of count
    _, _, split_mode = GPU_CONFIGS.get(gpu_config, (None, None, None))
    if split_mode == "tensor":
        n = sum(1 for mib in (peak_vram_mib or {}).values() if mib > 500)
        return GPU_BW_PEAK_GBS * max(n, 1)
    return GPU_BW_PEAK_GBS

def _add_bandwidth(r, gguf_size_gb, gpu_config=""):
    if r.get("ok") and gguf_size_gb and r.get("decode_tok_per_s"):
        bw   = gguf_size_gb * r["decode_tok_per_s"]
        peak = _effective_bw_peak(gpu_config, r.get("peak_vram_mib"))
        r["gguf_size_gb"]   = gguf_size_gb
        r["bandwidth_gb_s"] = bw
        r["bandwidth_pct"]  = bw / peak * 100
    return r

# ── Ollama helpers ────────────────────────────────────────────────────────────
def _ollama_unload(tag):
    subprocess.run(["ollama", "stop", tag], capture_output=True)
    for _ in range(30):
        time.sleep(0.5)
        try:
            ps = requests.get(f"{OLLAMA_API}/api/ps", timeout=3).json()
            if tag not in [m["name"] for m in ps.get("models", [])]:
                return
        except Exception:
            return

def _ollama_registered(tag):
    try:
        r = requests.post(f"{OLLAMA_API}/api/show", json={"name": tag}, timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        sys.exit("Ollama is not running — start it with: ollama serve")

# ── Ollama benchmark ──────────────────────────────────────────────────────────
def bench_ollama(m, prompt, max_tokens, tier=""):
    tag = m["ollama_tag"]
    payload = {"model": tag, "prompt": prompt, "stream": False,
               "keep_alive": "10m",
               "options": {"num_predict": max_tokens, "temperature": 0}}

    print(f"    warmup ...", end=" ", flush=True)
    r = requests.post(f"{OLLAMA_API}/api/generate", json=payload, timeout=600)
    if r.status_code != 200:
        return {"ok": False, "error": r.text}
    load_s = r.json().get("load_duration", 0) / 1e9
    print(f"loaded in {load_s:.1f}s")

    vram = VramSampler(); vram.start()
    pwr  = PowerSampler(); pwr.start()
    runs = []
    for i in range(REPEATS):
        print(f"    run {i+1}/{REPEATS} ...", end=" ", flush=True)
        r = requests.post(f"{OLLAMA_API}/api/generate", json=payload, timeout=600)
        d = r.json()
        dtps = d["eval_count"] / (d["eval_duration"] / 1e9)
        ptps = d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9)
        ttft = d["prompt_eval_duration"] / 1e9
        print(f"{dtps:.1f} tok/s  (prompt {ptps:.0f} tok/s  TTFT {ttft:.2f}s)")
        save_response(m["id"], "ollama", "auto", tier, i + 1, prompt, d.get("response", ""))
        runs.append({"decode_tok_per_s": dtps, "prompt_tok_per_s": ptps,
                     "ttft_s": ttft, "n_generated": d["eval_count"]})
    vram.stop(); pwr.stop()
    _ollama_unload(tag)

    avg = lambda k: sum(x[k] for x in runs) / len(runs)
    return {"ok": True, "load_time_s": load_s, "peak_vram_mib": vram.peak,
            "peak_watts": pwr.peak, "avg_watts": pwr.avg,
            "power_gpus": sorted(pwr.supported), "power_n": pwr.n_gpus,
            "decode_tok_per_s": avg("decode_tok_per_s"),
            "prompt_tok_per_s": avg("prompt_tok_per_s"),
            "ttft_s": avg("ttft_s"), "n_generated": runs[0]["n_generated"]}

# ── llama-server context manager ──────────────────────────────────────────────
@contextmanager
def llama_server(gguf_path, gpu_config, mtp_n=0, mtp_pmin=0.75):
    if not LLAMACPP_BIN.exists():
        raise FileNotFoundError(f"llama-server.exe not found at {LLAMACPP_BIN}")

    visible, tensor_split, split_mode = GPU_CONFIGS[gpu_config]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    cmd = [str(LLAMACPP_BIN), "-m", str(gguf_path), "-ngl", "-1",
           "--port", str(LLAMACPP_PORT), "--host", "127.0.0.1",
           "-c", str(N_CTX), "--flash-attn", "on", "-ub", "2048", "--log-disable"]
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

    log_path = Path(tempfile.mktemp(suffix=".log"))
    stderr_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=stderr_file)

    try:
        health = f"http://127.0.0.1:{LLAMACPP_PORT}/health"
        started = False
        for tick in range(360):
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
            proc.kill()
            raise TimeoutError("llama-server did not become ready within 180s")
        yield
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

# ── llama.cpp benchmark ───────────────────────────────────────────────────────
def bench_llamacpp(m, gpu_config, prompt, max_tokens, mtp_n=0, mtp_pmin=0.75, tier=""):
    url = f"http://127.0.0.1:{LLAMACPP_PORT}/v1/chat/completions"
    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": False}
    warmup  = {"messages": [{"role": "user", "content": "hi"}],
               "max_tokens": 4, "temperature": 0, "stream": False}

    # Evict Ollama's cached model so it doesn't hold VRAM when llama-server starts
    subprocess.run(["ollama", "stop", m["ollama_tag"]],
                   capture_output=True, timeout=15)
    time.sleep(1)

    mtp_tag = f" +mtp{mtp_n}" if mtp_n > 0 else ""
    print(f"    starting llama-server ({gpu_config}{mtp_tag}) ...", end=" ", flush=True)
    with llama_server(m["gguf"], gpu_config, mtp_n=mtp_n, mtp_pmin=mtp_pmin):
        print("ready")
        requests.post(url, json=warmup, timeout=300)
        vram = VramSampler(); vram.start()
        pwr  = PowerSampler(); pwr.start()
        runs = []
        for i in range(REPEATS):
            print(f"    run {i+1}/{REPEATS} ...", end=" ", flush=True)
            r = requests.post(url, json=payload, timeout=600)
            d = r.json(); t = d.get("timings", {})
            dtps = t.get("predicted_per_second", 0)
            ptps = t.get("prompt_per_second", 0)
            ttft = t.get("prompt_ms", 0) / 1000
            print(f"{dtps:.1f} tok/s  (prompt {ptps:.0f} tok/s  TTFT {ttft:.2f}s)")
            msg     = d.get("choices", [{}])[0].get("message", {})
            thinking = msg.get("reasoning_content", "")
            content  = msg.get("content", "") or ""
            response_text = (f"<think>\n{thinking}\n</think>\n\n{content}".strip()
                             if thinking else content)
            save_response(m["id"], "llamacpp", gpu_config, tier, i + 1, prompt, response_text)
            runs.append({"decode_tok_per_s": dtps, "prompt_tok_per_s": ptps,
                         "ttft_s": ttft,
                         "n_generated": d.get("usage", {}).get("completion_tokens", 0)})
        vram.stop(); pwr.stop()

    avg = lambda k: sum(x[k] for x in runs) / len(runs)
    return {"ok": True, "load_time_s": None, "peak_vram_mib": vram.peak,
            "peak_watts": pwr.peak, "avg_watts": pwr.avg,
            "power_gpus": sorted(pwr.supported), "power_n": pwr.n_gpus,
            "decode_tok_per_s": avg("decode_tok_per_s"),
            "prompt_tok_per_s": avg("prompt_tok_per_s"),
            "ttft_s": avg("ttft_s"), "n_generated": runs[0]["n_generated"]}

# ── Print helpers ─────────────────────────────────────────────────────────────
def print_result(r):
    if r.get("ok"):
        print(f"    decode  {r['decode_tok_per_s']:>7.1f} tok/s")
        print(f"    prompt  {r['prompt_tok_per_s']:>7.0f} tok/s  TTFT {r['ttft_s']:.2f}s")
        if r.get("bandwidth_gb_s"):
            print(f"    bw      {r['bandwidth_gb_s']:>7.1f} GB/s  ({r['bandwidth_pct']:.1f}% of peak)")
        if r.get("peak_watts"):
            sup = r.get("power_gpus") or []
            n   = r.get("power_n", len(sup))
            note = "" if len(sup) >= n else f"  [GPU {','.join(map(str, sup))} only — others report N/A]"
            print(f"    power   {r['avg_watts']:>6.1f}W avg  {r['peak_watts']:.1f}W peak  "
                  f"({r['avg_watts']/max(r['decode_tok_per_s'],0.001):.2f} W/tok·s⁻¹){note}")
        for gpu, mib in sorted((r.get("peak_vram_mib") or {}).items()):
            print(f"    GPU {gpu}   {mib/1024:>5.1f} GiB")
    else:
        print(f"    FAILED: {r.get('error', '?')}")

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
            print(f"[{i}] {name}")
            print(f"    {used:.1f} / {total:.1f} GiB  [{bar:<20}]")
    except Exception as e:
        print(f"pynvml error: {e}")


def cmd_models(_args):
    print(f"\n  {'id':<24} {'gguf file':<42} {'size':>6}  {'ollama'}")
    print("  " + "─" * 80)
    for m in MODELS:
        exists = m["gguf"].exists()
        size   = f"{m['gguf'].stat().st_size/1e9:.1f}G" if exists else "—"
        reg    = "✓" if _ollama_registered(m["ollama_tag"]) else "—"
        mark   = "✓" if exists else "✗"
        print(f"  {m['id']:<24} {mark} {m['gguf'].name:<40} {size:>6}  {reg}")
    print()


def cmd_register(args):
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
        r = subprocess.run(["ollama", "create", tag, "-f", str(mf)],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0:
            print(f"  {m['id']}  registered as {tag} ✓")
        else:
            print(f"  {m['id']}  FAILED:\n{r.stderr.strip()}")


def cmd_bench(args):
    m          = get_model(args.id)
    backends   = ["ollama", "llamacpp"] if args.backend == "both" else [args.backend]
    tiers      = args.tiers or list(BENCH_PROMPTS)
    gpu_cfgs   = args.gpu_configs or list(DEFAULT_GPU_CONFIGS)
    gguf_size  = m["gguf"].stat().st_size / 1e9 if m["gguf"].exists() else None

    label = "_".join([m["id"]] + (tiers if len(tiers) < 3 else []) +
                     (["_".join(gpu_cfgs)] if args.backend != "ollama" else []))
    with _log_session(label):
        _cmd_bench_inner(m, backends, tiers, gpu_cfgs, gguf_size, mtp_n=getattr(args, "mtp_n", 0))


def _cmd_bench_inner(m, backends, tiers, gpu_cfgs, gguf_size, mtp_n=0):
    _title = f"{m['id']}  —  {m['name']}"
    print(f"\n{'='*max(55,len(_title)+4)}\n  {_title}\n{'='*max(55,len(_title)+4)}")

    for tier in tiers:
        prompt     = BENCH_PROMPTS[tier]
        max_tokens = MAX_TOKENS_BY_TIER[tier]
        print(f"\n── tier: {tier} ──")

        if "ollama" in backends:
            if not _ollama_registered(m["ollama_tag"]):
                print(f"  [ollama] not registered — run: python run.py register {m['id']}")
            else:
                print(f"\n  [ollama]")
                r = _add_bandwidth(bench_ollama(m, prompt, max_tokens, tier=tier), gguf_size, "auto")
                r["gpu_config"] = "auto"
                print_result(r)
                save_result({**m, "backend": "ollama", "prompt_tier": tier, "mtp_n": 0, **r})

        if "llamacpp" in backends:
            if not m["gguf"].exists():
                print(f"  [llamacpp] gguf not found: {m['gguf']}")
            else:
                for cfg in gpu_cfgs:
                    mtp_tag = f" +mtp{mtp_n}" if mtp_n > 0 else ""
                    print(f"\n  [llamacpp | {cfg}{mtp_tag}]")
                    try:
                        r = _add_bandwidth(bench_llamacpp(m, cfg, prompt, max_tokens, mtp_n=mtp_n, tier=tier), gguf_size, cfg)
                        r["gpu_config"] = cfg
                    except (TimeoutError, RuntimeError, FileNotFoundError) as e:
                        print(f"    → {e}")
                        r = {"ok": False, "error": str(e), "gpu_config": cfg}
                    print_result(r)
                    save_result({**m, "backend": "llamacpp", "prompt_tier": tier, "mtp_n": mtp_n, **r})

    print(f"\n  results → {RESULTS_CSV}")


def cmd_run_all(args):
    tiers    = args.tiers or list(BENCH_PROMPTS)
    gpu_cfgs = args.gpu_configs or list(DEFAULT_GPU_CONFIGS)
    backends = ["ollama", "llamacpp"] if args.backend == "both" else [args.backend]
    label    = "run-all"
    with _log_session(label):
        for m in MODELS:
            gguf_size = m["gguf"].stat().st_size / 1e9 if m["gguf"].exists() else None
            _cmd_bench_inner(m, backends, tiers, gpu_cfgs, gguf_size, mtp_n=getattr(args, "mtp_n", 0))


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
    print(f"\n  {'id':<24} {'tier':<9} {'backend':<10} {'gpu':<9}{mtp_hdr}"
          f" {'decode t/s':>10} {'prompt t/s':>11} {'TTFT s':>7} {'bw%':>6}")
    print("  " + "─" * (92 + (5 if has_mtp else 0)))
    for r in ok:
        def f(k, fmt):
            try: return fmt.format(float(r[k]))
            except: return f"{'—':>8}"
        mtp_col = f" {r.get('mtp_n','0'):>4}" if has_mtp else ""
        print(f"  {r.get('id','?'):<24} {r.get('prompt_tier','?'):<9}"
              f" {r.get('backend','?'):<10} {r.get('gpu_config','?'):<9}{mtp_col}"
              f" {f('decode_tok_per_s','{:>10.1f}')}"
              f" {f('prompt_tok_per_s','{:>11.0f}')}"
              f" {f('ttft_s','{:>7.3f}')}"
              f" {f('bandwidth_pct','{:>5.1f}%')}")

    print(f"\n  {len(ok)} results  |  full analysis: python analyze.py")

# ── Argument parser ───────────────────────────────────────────────────────────
def main():
    p   = argparse.ArgumentParser(description="Dual-GPU LLM benchmark",
                                  formatter_class=argparse.RawDescriptionHelpFormatter,
                                  epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gpus",    help="Show GPU VRAM state")
    sub.add_parser("models",  help="List models and file status")
    sub.add_parser("results", help="Print results table")

    pr = sub.add_parser("register", help="Register model(s) with Ollama")
    pr.add_argument("id", nargs="?", help="Model id (omit for all)")

    pb = sub.add_parser("bench", help="Benchmark one model")
    pb.add_argument("id", help="Model id  (python run.py models to list)")
    pb.add_argument("--backend",     choices=["ollama", "llamacpp", "both"], default="both")
    pb.add_argument("--tiers",       nargs="+", choices=list(BENCH_PROMPTS),
                    help="Prompt tiers to run (default: all)")
    pb.add_argument("--gpu-configs", nargs="+", choices=list(GPU_CONFIGS),
                    dest="gpu_configs",
                    help="llama.cpp GPU configs (default: single0 single1 dual)")
    pb.add_argument("--mtp",         type=int, default=0, dest="mtp_n",
                    metavar="N",
                    help="MTP draft tokens (0=disabled; try 3–5 for ~2× decode speed)")

    pa = sub.add_parser("run-all", help="Benchmark all models")
    pa.add_argument("--backend",     choices=["ollama", "llamacpp", "both"], default="both")
    pa.add_argument("--tiers",       nargs="+", choices=list(BENCH_PROMPTS))
    pa.add_argument("--gpu-configs", nargs="+", choices=list(GPU_CONFIGS),
                    dest="gpu_configs")
    pa.add_argument("--mtp",         type=int, default=0, dest="mtp_n",
                    metavar="N",
                    help="MTP draft tokens (0=disabled)")

    args = p.parse_args()
    {"gpus": cmd_gpus, "models": cmd_models, "register": cmd_register,
     "bench": cmd_bench, "run-all": cmd_run_all, "results": cmd_results}[args.cmd](args)

if __name__ == "__main__":
    main()
