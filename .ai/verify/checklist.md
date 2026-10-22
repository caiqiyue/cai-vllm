# Verification Checklist

- The `python-cuda-inference-library` profile is recorded.
- Exactly one feature is `in_progress`.
- Both feature-list copies are identical.
- Scope and non-goals are written down.
- `bash ./init.sh` or `./init.ps1` ran.
- GPU-sensitive changes have real CUDA evidence; a `SKIP` is not accepted as GPU proof.
- Evidence is linked before a feature becomes `passing`.

