# feat-002 Prefix Cache verification — 2026-10-25

## Command

```bash
/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python scripts/prefix_cache_smoke.py
```

The script fixes GPU 0 and the PyTorch Attention backend internally. It generates one prompt, keeps its completed blocks in the BlockManager cache, then checks and generates a second prompt with the same 300-token prefix and a different suffix.

## Result

- Server commit: `0a39b12`
- GPU: physical GPU 0, NVIDIA GeForce RTX 2080 Ti
- `reusable_prefix_blocks=1`
- First and second generations both completed successfully.

This verifies a real Prefix Cache reuse path. FlashAttention, CUDA Graph, Tensor Parallel, preemption, and chunked-prefill stress paths remain outside this run.
