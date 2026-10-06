"""
Shared configuration: machine paths (config.toml / env vars) and the model
registry (models.toml). Imported by run.py and generate_report.py so adding a
model is a one-file change.
"""

import os
import shutil
import sys
import tomllib
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent   # repo root (src/ lives under it)
CONFIG_TOML   = ROOT / "config.toml"
MODELS_TOML   = ROOT / "models.toml"

RESULTS_DIR   = ROOT / "results"
RESULTS_CSV   = RESULTS_DIR / "results.csv"
LOGS_DIR      = RESULTS_DIR / "logs"
METRICS_DIR   = RESULTS_DIR / "metrics"
RUNS_DIR      = RESULTS_DIR / "runs"        # per-repetition rows + environment manifests
RESPONSES_DIR = RESULTS_DIR / "responses"
MODEL_INFO_JSON = RESULTS_DIR / "model_info.json"

# ── Machine config: config.toml < env vars ────────────────────────────────────
def _load_config():
    cfg = {}
    if CONFIG_TOML.exists():
        cfg = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    for k in ("GGUF_DIR", "LLAMACPP_BIN", "LLAMACPP_PORT", "OLLAMA_API", "GPU_BW_PEAK_GBS"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg

_cfg = _load_config()

def _default_llamacpp_bin():
    found = shutil.which("llama-server")
    return Path(found) if found else Path("llama-server")

GGUF_DIR        = Path(_cfg.get("GGUF_DIR", ROOT / "models"))
LLAMACPP_BIN    = Path(_cfg["LLAMACPP_BIN"]) if "LLAMACPP_BIN" in _cfg else _default_llamacpp_bin()
LLAMACPP_PORT   = int(_cfg.get("LLAMACPP_PORT", 8080))
OLLAMA_API      = _cfg.get("OLLAMA_API", "http://localhost:11434")
GPU_BW_PEAK_GBS = float(_cfg.get("GPU_BW_PEAK_GBS", 672.0))
OLLAMA_BIN      = shutil.which("ollama")    # None → Ollama backend unavailable

# ── Model registry ────────────────────────────────────────────────────────────
def load_models():
    if not MODELS_TOML.exists():
        sys.exit(f"Missing {MODELS_TOML}")
    entries = tomllib.loads(MODELS_TOML.read_text(encoding="utf-8")).get("model", [])
    models, seen = [], set()
    for e in entries:
        for k in ("id", "name", "file"):
            if k not in e:
                sys.exit(f"models.toml: entry {e} is missing '{k}'")
        if e["id"] in seen:
            sys.exit(f"models.toml: duplicate id '{e['id']}'")
        seen.add(e["id"])
        models.append({
            "id":         e["id"],
            "name":       e["name"],
            "gguf":       GGUF_DIR / e["file"],
            "ollama_tag": e.get("ollama_tag", f"eval/{e['id']}"),
            "moe":        e.get("moe"),        # None → resolved from GGUF metadata
            "include":    e.get("include", True),  # false → run-all skips unless named
            "source":     e.get("source", ""),
        })
    return models

MODELS = load_models()
