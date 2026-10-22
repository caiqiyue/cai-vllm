# Feature Delivery Workflow

1. Planner defines one feature, scope, non-goals, and executable verification.
2. Implementer changes only that feature.
3. Reviewer checks scope, API compatibility, and CUDA-risk boundaries.
4. Verifier runs required checks and records evidence.
5. State moves to `passing` only after evidence exists.

If scope is unclear, return to planning. If implementation or verification fails, return to the implementer. Never weaken checks to create a pass.

