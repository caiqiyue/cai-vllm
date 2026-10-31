# feat-002 GPU gate and example generation on server GPU 0 — 2026-10-31

## Environment

| Item | Value |
| --- | --- |
| Host | `k8smaster-empty` (SSH alias `paper3-server`) |
| GPU | physical GPU 0, NVIDIA GeForce RTX 2080 Ti (11264 MiB) |
| Python | `/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python` (3.11) |
| PyTorch | 2.7.1+cu126, `cuda=True` |
| Attention backend | `CAI_VLLM_ATTENTION_BACKEND=torch` |
| Model | `/mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507` |
| Repo | `/mnt/public/caiqiyue_file/cai-vllm` at `de0114a` |

## Commands

```bash
cd /mnt/public/caiqiyue_file/cai-vllm
git pull --ff-only origin main

export CAI_VLLM_PYTHON=/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python
CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh

/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python example.py \
  /mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507 \
  --max-model-len 512 --gpu-memory-utilization 0.85 --max-tokens 32
```

`CAI_VLLM_PYTHON` is required on this host: outside the conda environment there is no `python`
on `PATH`, so the harness used to fail at L0 with `python: 未找到命令`. `init.sh` now resolves
`python` -> `python3` and honours `CAI_VLLM_PYTHON`.

## Results

GPU gate (all levels passed):

```text
[L0] Environment ... [PASS] Python 3.11
[L1] Harness state ... [PASS] one active feature: feat-002
[L2] Python syntax and package metadata ... [PASS]
[L3] Repository hygiene ... [PASS] no whitespace errors
[L4] Runtime dependency and import smoke
[PASS] torch=2.7.1+cu126 cuda=True
```

`example.py` on the real Qwen3-4B checkpoint returned generated text without runtime errors:

```text
Prompt: '<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n'
Completion: "Hi there! 👋 I'm Qwen, a large-scale language model developed by Alibaba Cloud's Tongyi Lab. I'm really excited to be here and"

Prompt: '<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n'
Completion: 'The prime numbers within 100 are:\n\n**2, 3, 5, 7, 11, 13, 1'
```

(The second completion is truncated at `--max-tokens 32`, which is the configured limit, not a
failure.)

Unit tests on the same host: `Ran 5 tests ... OK`.

## Prefix cache (same run, same commit)

```text
block_size=256
cold: prompt_tokens=262 prefill_tokens=262 cached_tokens=0
warm: prompt_tokens=262 prefill_tokens=6 cached_tokens=256
saved_prefill_tokens=256
hash_index_blocks=1
prefix_cache_hit=YES
```

The warm request reused 256 cached tokens and prefilled 6 tokens instead of 262. Full detail in
`evidence/feat-002-2026-10-31-prefix-cache-engine-level.md`.

## Still outside this run

FlashAttention (install/build unresolved), CUDA Graph (disabled by `enforce_eager=True`),
Tensor Parallel, scheduler preemption, chunked-prefill stress, and long-context pressure.
