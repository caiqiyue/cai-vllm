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

## GPU verification (server, GPU 0) — DONE 2026-10-31

```bash
cd /mnt/public/caiqiyue_file/cai-vllm
git pull --ff-only origin main
/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python scripts/prefix_cache_smoke.py
```

Measured output on the RTX 2080 Ti (GPU 0):

```text
block_size=256
cold: prompt_tokens=262 prefill_tokens=262 cached_tokens=0
warm: prompt_tokens=262 prefill_tokens=6 cached_tokens=256
saved_prefill_tokens=256
hash_index_blocks=1
prefix_cache_hit=YES
```

The warm request reused 256 cached tokens and prefilled 6 tokens instead of 262, so the engine
really hits the prefix cache rather than only holding a hash-index entry. Companion evidence:
`evidence/feat-002-2026-10-31-server-gpu0-gate.md`.

FlashAttention, CUDA Graph, Tensor Parallel, preemption, and chunked-prefill stress paths remain
outside this run.
