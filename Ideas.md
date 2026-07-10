Ideas to improve the product
### Pending

- ub (batch size) checks
- Thinking on vs off — needs a real toggle mechanism decided per backend/model before it can be a model tag (see conversation with Claude)

### Implemented

- Flash attention on/off — `bench`/`run-all --flash-attn {on,off}` (default on)
- mmap on/off — `bench`/`run-all --mmap {on,off}` (default on)
- Offload everything to GPU (`ngl 999` equivalent) — already `-ngl -1` in `run.py`