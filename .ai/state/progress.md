# Structured Progress

The detailed session log is maintained in `claude-progress.md`.

- Current active feature: `feat-002`
- Selected profile: `python-cuda-inference-library`
- Standard verification: `bash ./init.sh`
- Full GPU gate: `CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh`
- Verified feature: `feat-001` with evidence in `evidence/feat-001-2026-10-22.md`.
- Known blocker: local PyTorch DLL loading prevents `feat-002` runtime verification on this host.

