# Progress Log

## Current Verified State

- Repository root: `E:\ai-learning\07-inference-and-serving\cai-vllm`
- Standard startup path: `bash ./init.sh` or `./init.ps1`
- Standard lightweight verification path: `bash ./init.sh`
- Full GPU gate: `CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh`
- Current active feature: `feat-002` — Validate CUDA inference journey
- Current blocker: local PyTorch fails to load `torch_python.dll`; GPU/runtime smoke must be verified on a supported environment.
- Learner-facing notes: `docs/nano-vllm-inference-flow.md` documents the full inference chain and per-component mechanisms (`feat-004`, passing).

## Session Log

### Session 001 — Repository and harness bootstrap

- Date: 2026-10-22
- Goal: remove inherited Git history, configure the new CAI-vLLM repository, install the L2 harness, and refresh README.
- Baseline observation: the checkout pointed to `GeeeekExplorer/nano-vllm`; source files had been moved to the non-importable `vllm-src/` directory.
- Scope: repository metadata, source-package directory restoration, harness artifacts, verification entry points, and documentation.
- Verification: `./init.ps1` and `bash ./init.sh` both exited 0; evidence is recorded in `evidence/feat-001-2026-10-22.md`.
- Completed: repository identity, importable package layout, L2 harness, README, and cross-platform lightweight verification.
- Known environment issue: installed PyTorch cannot load on this Windows/Python environment; the harness reports this as an explicit `SKIP` unless GPU verification is required.

### Session 002 — README learning-repository update

- Date: 2026-10-22
- Goal: reposition README as source-code-first with `docs/` as supporting AI Infra inference notes.
- Completed: removed the legacy logo and benchmark promotion, added the code map and reading path, and created `docs/README.md` as the learning-note index.
- Repository policy: `docs/` contains learning content only; development design and execution artifacts remain untracked.
- Verification: both Harness entry points passed control checks; all local Markdown links resolve; evidence is in `evidence/feat-003-2026-10-22.md`.
- Next: `feat-002` remains the sole active feature and requires supported CUDA hardware.

### Session 003 — Server GPU 0 deployment attempt

- Date: 2026-10-25
- Server repository synchronized from GitHub at commit `0cf8279`.
- Created `/mnt/public/caiqiyue_file/conda-envs/cai-vllm` with Python 3.11.
- Verified PyTorch `2.7.1+cu126` CUDA smoke on physical GPU 0 (`NVIDIA GeForce RTX 2080 Ti`) with `CUDA_VISIBLE_DEVICES=0`.
- Installed Triton, Transformers, NumPy, and xxhash.
- Full CAI-vLLM import/generation remains unverified because FlashAttention wheel compilation stalled; evidence is in `evidence/feat-002-2026-10-25-server-gpu0.md`.
- Added and verified the PyTorch Attention fallback: two unit tests, the required GPU gate, and a real Qwen3-4B generation succeeded on GPU 0 without FlashAttention. FlashAttention remains an optional unverified acceleration path; evidence is in `evidence/feat-002-2026-10-25-torch-fallback.md`.

### Session 004 — Prefix Cache and handoff alignment

- Date: 2026-10-25
- Added `scripts/prefix_cache_smoke.py`, which generated two prompts with a 300-token shared prefix and reported `reusable_prefix_blocks=1` on GPU 0.
- Documented the runtime verification matrix in `harness.md`, `README.md`, and the handoff artifacts.
- Full current script coverage and remaining unverified paths are recorded in `evidence/feat-002-2026-10-25-prefix-cache.md`.

### Session 005 — Inference flow learning note

- Date: 2026-10-29
- Goal: turn the completed read-through of the `torch_fallback_smoke.py` debug session into a durable learning artifact.
- Added `docs/nano-vllm-inference-flow.md`: component map, ten-stage breakdown, ten per-component mechanism chapters, tensor/shape contract table, constant provenance table, a frame-by-frame walkthrough of the smoke-script parameters, six typical execution paths, a failure-mode table, and an explicit pending-verification list.
- Registered the note as the shared entry point in `docs/README.md`, `README.md`, and `harness.md`.
- Recorded the work as `feat-004` in both feature-list copies; `feat-002` remains the only `in_progress` feature and its blocker is unchanged.
- Verification: local Markdown link check over `docs/**/*.md` and the root `README.md` reported 0 broken links; `bash ./init.sh` passed L0–L3 and reported an explicit `SKIP` for the GPU/runtime smoke on this host. Evidence is in `evidence/feat-004-2026-10-29-inference-flow-doc.md`.
- Known limitation recorded in the note: unverified server-side numbers (26 KV blocks, prompt lengths 12/7) and the flash-vs-torch numerical equivalence are marked as pending verification rather than stated as fact.
- Reviewer note: the note's depth and structure were judged unsatisfactory by the requester on first delivery; it is committed as a v1 baseline to be revised, not as a finished study.

