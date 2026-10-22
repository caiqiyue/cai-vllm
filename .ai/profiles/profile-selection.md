# Profile Selection

- Selected profile: `python-cuda-inference-library`
- Base harness profile: `generic-full-stack-ai-app` adapted for a Python/CUDA library

## Evidence

- `pyproject.toml` defines a Python 3.10-3.12 package.
- `nanovllm/` contains model, attention, scheduler, KV-cache, and tensor-parallel code.
- Dependencies include PyTorch, Triton, Transformers, FlashAttention, and xxhash.
- `example.py` and `bench.py` are the public inference and benchmark journeys.

## Verification Surfaces

- Harness state consistency and WIP=1.
- Python syntax and package metadata.
- Import smoke with runtime dependencies.
- Real model generation on supported CUDA hardware.
- Benchmark behavior when performance-sensitive code changes.

