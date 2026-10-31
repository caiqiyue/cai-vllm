# feat-002 Prefix Cache engine-level verification — 2026-10-31

## Problem found

`scripts/prefix_cache_smoke.py` reported `reusable_prefix_blocks=1` and passed, but it never
observed the engine. It built a throw-away `Sequence` and called
`BlockManager.can_allocate()` **before** the second request, then asserted that the returned
number was positive.

That proves only that the hash index contains an entry for the shared prefix. It does not prove
that the second request actually reused the KV block: the real reuse happens later, inside
`Scheduler.schedule()` → `BlockManager.allocate()`, and the script never inspected it. A cache
that is populated but never reused, or reused with the wrong token count, would still have
produced `reusable_prefix_blocks=1` and a green script.

## Change

`scripts/prefix_cache_smoke.py` rewritten (no `nanovllm/` source changed):

- Prompts are built at token level, so the first `block_size` token ids are byte-identical for
  both requests and the shared part is exactly one full (hashable) block.
- `observe_scheduler()` wraps `llm.scheduler.schedule` in the script and records, per step, the
  engine's real `num_cached_tokens` and `num_scheduled_tokens`.
- `summarize_prefix_cache()` (pure function, no GPU needed) asserts the engine-level contract:
  - cold request: `cached_tokens == 0` and prefill == full prompt length;
  - warm request: `cached_tokens >= block_size` and prefill == prompt length − block_size;
  - the warm request must prefill strictly fewer tokens than the cold one.
  Any violation raises a `RuntimeError` naming the failing condition.
- Output now reports `saved_prefill_tokens` and `prefix_cache_hit=YES`.

New CPU-only unit test: `tests/test_prefix_cache_smoke.py` (4 cases, no CUDA required) covering
the shared-prefix construction, a real hit, a missed hit, and a partial reuse.

## Local verification (this host, no CUDA)

```bash
python -m unittest tests.test_prefix_cache_smoke tests.test_torch_fallback_smoke -v
python -m py_compile scripts/prefix_cache_smoke.py tests/test_prefix_cache_smoke.py
bash ./init.sh
```

- `Ran 5 tests ... OK` (4 new prefix-cache cases + the existing environment case).
- `COMPILE_OK`.
- Harness: L0–L3 `[PASS]`, L4 `[SKIP] GPU/runtime smoke unavailable in this environment`
  (local PyTorch cannot load `torch_python.dll`; unchanged known blocker).

## Pending GPU verification (server, GPU 0)

```bash
cd /mnt/public/caiqiyue_file/cai-vllm
/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python scripts/prefix_cache_smoke.py
```

Expected output shape: `cold prefill_tokens = N`, `warm cached_tokens = 256`,
`warm prefill_tokens = N-256`, `saved_prefill_tokens = 256`, `prefix_cache_hit=YES`.

This run has **not** been executed yet; the engine-level numbers above are code-derived, not
measured. FlashAttention, CUDA Graph, Tensor Parallel, preemption, and chunked-prefill stress
paths remain outside this run.
