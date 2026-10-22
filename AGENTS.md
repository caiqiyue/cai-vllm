# CAI-vLLM Agent Guide

CAI-vLLM is an educational, lightweight vLLM implementation for studying modern LLM inference internals. Agents MUST optimize for correctness, reproducibility, and readable learning-oriented code before speed or feature count.

Identity check: your working name is `CaiVLLM-Agent`; the project codename is `CAI-vLLM`.

## Non-negotiable rules

- MUST keep work-in-progress to exactly one active feature.
- MUST NOT claim completion without recorded, reproducible verification evidence.
- MUST preserve the public Python import path `nanovllm` unless an active feature explicitly changes it.
- MUST keep CUDA/Triton fast paths intact; CPU-only environments may report an explicit `SKIP`, never a silent pass.
- MUST NOT run long model downloads or GPU benchmarks during routine startup verification.
- MUST NOT commit model weights, generated caches, secrets, or local virtual environments.
- MUST keep changes inside the active feature and avoid opportunistic refactors.
- MUST update state and handoff artifacts before ending a session.

## Startup workflow

Before implementation:

1. Confirm the repository root with `pwd` or `Get-Location`.
2. Read `claude-progress.md` and `session-handoff.md`.
3. Read `feature_list.json` and confirm exactly one feature is `in_progress`.
4. Read `.ai/policies/global-rules.md` and `.ai/policies/definition-of-done.md`.
5. Review recent commits with `git log --oneline -5`.
6. Run `bash ./init.sh` (or `./init.ps1` on PowerShell).

If the baseline fails, repair or document that failure before starting feature work.

## Working rules

- Treat `feature_list.json` as the human-facing source of truth and keep `.ai/state/feature-list.json` identical.
- Implement only the active feature. New ideas go into notes, not the current diff.
- Record commands and outcomes in `evidence/` immediately after verification.
- Distinguish `implemented` from `passing`; only evidence-backed work may become `passing`.
- Prefer small, readable inference components and explicit tensor-shape invariants.
- When changing scheduling, KV-cache, attention, tensor-parallel, or model-loading code, require a CUDA-capable validation path in addition to static checks.

## Required artifacts

- `feature_list.json`: canonical feature state machine.
- `.ai/state/feature-list.json`: checked mirror used by the structured harness.
- `claude-progress.md`: current verified state and append-only session log.
- `session-handoff.md`: next-session continuity card.
- `init.sh` / `init.ps1`: standard lightweight verification entry points.
- `evidence/`: reproducible verification logs.

## Definition of done

A feature is done only when all are true:

- Target behavior is implemented within the recorded scope.
- Relevant static, unit/integration, and GPU checks ran, or unavailable hardware was explicitly recorded as `SKIP`.
- Evidence is linked from both feature-list copies.
- README and harness documentation remain accurate.
- A fresh session can run the standard verification path without manual repair.

## Routing map

- `.ai/state/active-task.md`: concise active-task scope.
- `.ai/policies/`: global delivery rules and Definition of Done.
- `.ai/workflows/feature-delivery.md`: planner → implementer → reviewer → verifier workflow.
- `.ai/verify/`: verification expectations and role flow.
- `.ai/profiles/profile-selection.md`: detected repository profile and verification surfaces.
- `harness.md`: source methodology for this repository's harness.

## Verification commands

```bash
bash ./init.sh
CAI_VLLM_REQUIRE_GPU=1 bash ./init.sh
```

```powershell
./init.ps1
./init.ps1 -RequireGpu
```

## End of session

1. Update `claude-progress.md`.
2. Update both feature-list copies with truthful status, evidence, and notes.
3. Update `session-handoff.md` with unresolved or unverified work.
4. Run the required verification and complete `clean-state-checklist.md`.
5. Commit with a message that references the feature ID.

## Context pressure protocol

- At 50%, record important decisions as they are made.
- At 70%, stop opening new work and prepare the handoff.
- At 85%, run the end-of-session sequence and leave a clean commit.
- At 95%, write a concise emergency handoff and stop expanding scope.

