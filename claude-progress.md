# Progress Log

## Current Verified State

- Repository root: `E:\ai-learning\07-inference-and-serving\cai-vllm`
- Standard startup path: `bash ./init.sh` or `./init.ps1`
- Standard lightweight verification path: `bash ./init.sh`
- Full GPU gate: `CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh`
- Current active feature: `feat-002` — Validate CUDA inference journey
- Current blocker: local PyTorch fails to load `torch_python.dll`; GPU/runtime smoke must be verified on a supported environment.

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

