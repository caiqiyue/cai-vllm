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
- Latest GitHub/server handoff commit: `a3b8fdf`
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

## Current Handoff Override — 2026-10-29

The sections above contain historical notes. Use this section as the current state.

### State changes this session

| Item | Before | After |
| --- | --- | --- |
| Active feature | `feat-002` | `feat-002` (unchanged, still `in_progress`) |
| Feature list | 3 features | 4 features (`feat-004` added, `passing`) |
| Learner notes | `docs/README.md` index only | `docs/nano-vllm-inference-flow.md` added and indexed |

`feat-002` blocker is unchanged: local PyTorch cannot load `torch_python.dll`, so the GPU/runtime smoke stays an explicit `SKIP` on this host.

### New artifact

- `docs/nano-vllm-inference-flow.md` — the nano-vLLM inference chain: component map, ten-stage breakdown, ten per-component mechanism chapters (`Sequence`, `Scheduler`, `BlockManager`, Paged KV Cache, `ModelRunner`/`Context`, Attention backends, memory budget, CUDA Graph, Tensor Parallel, sampling), tensor/shape contract table, constant provenance table, a frame-by-frame walkthrough of the `torch_fallback_smoke.py` parameters, six typical execution paths, a failure-mode table, and a pending-verification list.
- Registered in `docs/README.md`, root `README.md`, and `harness.md`.
- No `nanovllm/` source, test, or runtime script was changed.

### Unverified or disabled

- Server-side numbers quoted in the note (26 KV blocks, prompt lengths 12 and 7) come from the earlier script debug session and were not reproduced on this host; they are labeled as pending verification in the note.
- torch-vs-flash numerical equivalence and the flash paged-path `cu_seqlens_k` offset semantics remain code-derived conclusions only.
- FlashAttention installation/build is still unresolved.
- CUDA Graph is still disabled with `enforce_eager=True` in every script.
- Tensor Parallel and shared-memory worker paths are still untested; the scope stays GPU 0 only.
- Scheduler preemption, chunked-prefill stress, long-context pressure, and benchmark throughput are still uncovered.

### Known quality issue with the new note

The requester judged the first delivery of `docs/nano-vllm-inference-flow.md` unsatisfactory. It is committed as a v1 baseline, not as a finished study. The next session should revise it before treating it as the authoritative learning artifact.

### Next best step

- Highest-priority active feature: `feat-002`. Passing still requires a real Qwen3 generation run on supported NVIDIA hardware with evidence under `evidence/`.
- Secondary: revise `docs/nano-vllm-inference-flow.md` per the requester's feedback, and turn the unverified items in its section 10 into real server-side experiments.

### Reproduction commands

```bash
cd /mnt/public/caiqiyue_file/cai-vllm
/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python scripts/torch_fallback_smoke.py
/mnt/public/caiqiyue_file/conda-envs/cai-vllm/bin/python scripts/prefix_cache_smoke.py
CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh
```

```bash
cd /mnt/public/caiqiyue_file/cai-vllm
bash ./init.sh          # lightweight verification, L4 reports SKIP without CUDA
```

