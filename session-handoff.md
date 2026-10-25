# Session Handoff

## Verified Now

- `feat-001` passed the Windows and WSL lightweight harness checks; evidence is in `evidence/feat-001-2026-10-22.md`.
- `feat-003` passed documentation and link verification; evidence is in `evidence/feat-003-2026-10-22.md`.

## Changed This Session

- Restored the importable `nanovllm/` source-package path.
- Added project-specific harness instructions, state, workflows, and cross-platform verification entry points.
- Prepared package metadata and README for the `caiqiyue/cai-vllm` repository.
- Reframed README as source-code-first and added `docs/README.md` for learner-facing notes only.

## Broken Or Unverified

- Local PyTorch cannot load `torch_python.dll`, so CUDA inference is not verified on this host.
- Full Qwen3 model inference requires a supported CUDA environment and model weights.
- Server GPU 0 CUDA smoke and full short Qwen3-4B generation passed on 2026-10-25 using the new PyTorch Attention fallback; FlashAttention remains an optional unverified acceleration path.

## Next Best Step

- Highest-priority active feature: `feat-002`.
- Passing requires a real Qwen3 generation run on supported NVIDIA hardware with evidence recorded under `evidence/`.
- Do not weaken the lightweight checks or turn GPU failures into silent success.

## Commands

- Startup: `bash ./init.sh`
- Windows startup: `./init.ps1`
- Required GPU gate: `CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh`

## Server deployment state

- Repository: `/mnt/public/caiqiyue_file/cai-vllm` at `0cf8279`
- Environment: `/mnt/public/caiqiyue_file/conda-envs/cai-vllm`
- GPU restriction: `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0`
- Next step: if performance validation is needed, diagnose or provide a compatible FlashAttention build; ordinary generation is already runnable with `CAI_VLLM_ATTENTION_BACKEND=torch`.

## Current Handoff Override — 2026-10-25

The sections above contain historical notes. Use this section as the current state.

### Verified scripts

| Entry | Coverage |
| --- | --- |
| `scripts/torch_fallback_smoke.py` | LLM API, tokenizer, batch Prefill/Decode, Scheduler, Qwen3, PyTorch Attention, Paged KV Cache, sampling, output |
| `scripts/prefix_cache_smoke.py` | Prefix Cache reuse across two real requests; latest run reported `reusable_prefix_blocks=1` |
| `tests/test_attention_backend.py` | Prefill, Decode, GQA, PyTorch KV Cache write/read |
| `tests/test_torch_fallback_smoke.py` | Smoke-script environment configuration |

### Verified server state

- Repository: `/mnt/public/caiqiyue_file/cai-vllm`
- Latest synchronized commit before this handoff: `0a39b12`
- Environment: `/mnt/public/caiqiyue_file/conda-envs/cai-vllm`
- GPU: physical GPU 0, NVIDIA GeForce RTX 2080 Ti
- Backend: `CAI_VLLM_ATTENTION_BACKEND=torch`

### Unverified or disabled

- FlashAttention installation/build remains unresolved.
- CUDA Graph is disabled with `enforce_eager=True`.
- Tensor Parallel and shared-memory worker paths are not tested because the scope is GPU 0 only.
- Scheduler preemption, chunked-prefill stress, long-context pressure, and benchmark throughput are not covered.

### Reproduction commands

```bash
cd /mnt/public/caiqiyue_file/cai-vllm
/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python scripts/torch_fallback_smoke.py
/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python scripts/prefix_cache_smoke.py
CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh
```

