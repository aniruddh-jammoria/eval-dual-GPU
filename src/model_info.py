"""
Architecture facts read straight from GGUF headers (no weights loaded).

Used for:
  - MoE detection (expert_count > 0) instead of trusting names
  - bytes actually read per decoded token (M7): dense tensors in full, routed
    experts scaled by expert_used_count / expert_count, token_embd skipped
    (one row per token) unless it doubles as the output head
  - layer count / hidden size for the paper's communication-cost model

Results are cached in results/model_info.json keyed by filename + size + mtime.

Usage:  python src/model_info.py              print table for all registered models
"""

import json
import sys

from config import MODEL_INFO_JSON, MODELS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def _field(reader, key):
    f = reader.fields.get(key)
    if f is None or not f.data:
        return None
    v = f.parts[f.data[0]]
    try:
        if f.types and f.types[0].name == "STRING":
            return bytes(v).decode("utf-8", errors="replace")
        return v.tolist()[0] if hasattr(v, "tolist") else v
    except Exception:
        return None

def inspect_gguf(path):
    from gguf import GGUFReader
    r    = GGUFReader(str(path))
    arch = _field(r, "general.architecture") or ""
    a    = lambda k: _field(r, f"{arch}.{k}")

    n_expert      = a("expert_count") or 0
    n_expert_used = a("expert_used_count") or 0
    names         = {t.name for t in r.tensors}
    tied_output   = "output.weight" not in names

    total_bytes = active_bytes = 0
    total_params = active_params = 0
    for t in r.tensors:
        nb, ne = int(t.n_bytes), int(t.n_elements)
        total_bytes  += nb
        total_params += ne
        if t.name == "token_embd.weight" and not tied_output:
            continue                                  # lookup: one row per token
        frac = (n_expert_used / n_expert) if ("_exps" in t.name and n_expert) else 1.0
        active_bytes  += nb * frac
        active_params += ne * frac

    return {
        "file":               path.name,
        "architecture":       arch,
        "general_name":       _field(r, "general.name"),
        "file_type":          _field(r, "general.file_type"),
        "n_layers":           a("block_count"),
        "hidden_dim":         a("embedding_length"),
        "n_heads":            a("attention.head_count"),
        "n_kv_heads":         a("attention.head_count_kv"),
        "context_length":     a("context_length"),
        "expert_count":       n_expert,
        "expert_used_count":  n_expert_used,
        "moe":                n_expert > 0,
        # native decision models (/v1/systemone) carry {arch}.decision.type metadata
        "decision_type":      a("decision.type"),
        "tied_output":        tied_output,
        "total_params":       total_params,
        "active_params":      round(active_params),
        "total_bytes":        total_bytes,
        "active_bytes":       round(active_bytes),
    }

def _load_cache():
    try:
        return json.loads(MODEL_INFO_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_info(path):
    """Cached inspect_gguf(); returns None if the file is missing or unreadable."""
    if not path.exists():
        return None
    st    = path.stat()
    stamp = f"{st.st_size}:{int(st.st_mtime)}"
    cache = _load_cache()
    hit   = cache.get(path.name)
    if hit and hit.get("_stamp") == stamp:
        return hit
    try:
        info = inspect_gguf(path)
    except Exception as e:
        print(f"  [model_info] could not read {path.name}: {e}", file=sys.stderr)
        return None
    info["_stamp"] = stamp
    cache[path.name] = info
    MODEL_INFO_JSON.parent.mkdir(parents=True, exist_ok=True)
    MODEL_INFO_JSON.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return info

def resolve_moe(m, info):
    # explicit models.toml override wins; else GGUF metadata; else False
    if m.get("moe") is not None:
        return bool(m["moe"])
    return bool(info and info["moe"])

def main():
    print(f"\n  {'id':<22} {'arch':<12} {'layers':>6} {'hidden':>6} {'experts':>8}"
          f" {'params':>8} {'active':>8} {'GGUF GB':>8} {'read/tok GB':>11}")
    print("  " + "─" * 98)
    for m in MODELS:
        info = get_info(m["gguf"])
        if not info:
            print(f"  {m['id']:<22} (gguf missing)")
            continue
        exp = f"{info['expert_used_count']}/{info['expert_count']}" if info["moe"] else "—"
        print(f"  {m['id']:<22} {info['architecture']:<12} {info['n_layers'] or '?':>6}"
              f" {info['hidden_dim'] or '?':>6} {exp:>8}"
              f" {info['total_params']/1e9:>7.1f}B {info['active_params']/1e9:>7.1f}B"
              f" {info['total_bytes']/1e9:>8.2f} {info['active_bytes']/1e9:>11.2f}")
    print()

if __name__ == "__main__":
    main()
