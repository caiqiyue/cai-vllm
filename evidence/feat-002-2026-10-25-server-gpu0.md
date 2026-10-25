# feat-002 server GPU0 verification — 2026-10-25

## Scope

- Server: `k8smaster-empty`
- Repository: `/mnt/public/caiqiyue_file/cai-vllm`
- Conda environment: `/mnt/public/caiqiyue_file/conda-envs/cai-vllm`
- GPU policy: `CUDA_VISIBLE_DEVICES=0`; physical GPU 0 only

## Recorded results

- Local `main` and GitHub `origin/main` were already equal at `0cf8279`; no new local commit was required.
- Server repository was cloned from GitHub at commit `0cf8279` with a clean worktree.
- Conda Python `3.11.17` environment was created successfully.
- Installed: PyTorch `2.7.1+cu126`, Triton `3.3.1`, Transformers `5.20.2`, NumPy `2.4.6`, and xxhash `4.0.2`.
- With `CUDA_VISIBLE_DEVICES=0`, a CUDA matrix multiplication succeeded and reported one visible device: `NVIDIA GeForce RTX 2080 Ti`.
- CUDA toolkit compiler is available at `/usr/local/cuda-12.8/bin/nvcc`.

## Blocker

- `flash-attn` was attempted with the server CUDA toolkit and with `TORCH_CUDA_ARCH_LIST=7.5`, `FLASH_ATTENTION_CUDA_ARCHS=75`, and bounded build parallelism.
- The wheel build remained in `build_wheel` without producing `nvcc`/`ninja` compiler children or an installable package; the stalled build processes were stopped.
- `PYTHONPATH=/mnt/public/caiqiyue_file/cai-vllm python -c 'import nanovllm'` therefore fails at the required `flash_attn` import.
- Full Qwen3 generation was not run because the required FlashAttention dependency is unavailable. No model weights were downloaded.

## Reproduction commands

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python -c \
  'import torch; x=torch.randn((2048,2048), device=torch.device(0)); y=x@x; torch.cuda.synchronize(); print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), torch.cuda.get_device_name(0))'
```

