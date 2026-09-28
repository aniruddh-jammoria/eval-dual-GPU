Ideas to improve the product
### Pending

- Thinking on vs off — needs a real toggle mechanism decided per backend/model before it can be a model tag (see conversation with Claude)

### Implemented

- ub (batch size) checks — `--ubatch`, Exp. X1 (2026-09-29); default now `-ub 512`

- Flash attention on/off — `bench`/`run-all --flash-attn {on,off}` (default on)
- mmap on/off — `bench`/`run-all --mmap {on,off}` (default on)
- Offload everything to GPU (`ngl 999` equivalent) — already `-ngl -1` in `run.py`