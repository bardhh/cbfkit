# Bolt -> Develop Execution Playbook (PR-by-PR)

This document operationalizes `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_premerge_design_review.md` into a sequence that can be executed by an integration owner.

## Preconditions

1. Branch topology check:
   - `git rev-list --left-right --count develop...origin/bolt` expected `0 231`.
2. Git conflict check:
   - `git merge-tree $(git merge-base develop origin/bolt) develop origin/bolt` expected no conflict markers.
3. Attach evidence artifacts to integration PR:
   - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_commit_inventory.tsv`
   - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_premerge_design_review.md`

## Semantic Conflict Register

| ID | Conflict | Evidence (file:line) | Impact | Pre-merge action |
|---|---|---|---|---|
| SC-1 | Solver status interpretation drift | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/cbf_clf/cbf_clf_qp_generator.py:510`, `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/simulator.py:113` | Inconsistent safety/diagnostics | Align status map and warnings to strict success policy |
| SC-2 | New status `-2` (NaN input) not fully represented in simulator map | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/cbf_clf/cbf_clf_qp_generator.py:528`, `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/simulator.py:54` | Error reporting ambiguity | Add explicit mapping and regression tests |
| SC-3 | Public controller/signature contract shift | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/utils/user_types.py:235` | Downstream runtime/type break risk | Add compatibility adapters + migration notes |
| SC-4 | Simulation return API shift (`SimulationResults`) | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/utils/user_types.py:84` | Consumer API break risk | Validate tuple-compatible behavior and key-based access in integration tests |
| SC-5 | Dependency model split to optional extras | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/pyproject.toml:33` | Environment/profile mismatch | Add explicit CI jobs per dependency profile |
| SC-6 | Optional dependency runtime errors added | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/optimization/quadratic_program/qp_solver_casadi.py:41`, `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/utils/visualization.py:20` | Mode-specific runtime failures | Add smoke tests and clear failure guidance checks |
| SC-7 | Duplicated integrator fast-path logic in Python and JIT stacks | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/backend.py:157`, `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/simulator_jit.py:167` | Long-term divergence and regressions | Refactor to single shared integration helper |
| SC-8 | MPPI algorithm rewrite (scan/vectorization/sampling) | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/mppi/mppi_source.py:76` | Behavioral/perf drift risk | Benchmark + parity scenarios before merge |
| SC-9 | Observability stack migration (`tqdm` -> `rich`) | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/callbacks.py:3` | CI/non-TTY output behavior changes | Add CI log sanity checks |
| SC-10 | Tutorial entrypoint churn | `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/README.md:118`, `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/tests/test_examples_and_tutorials.py:17`, deleted `tutorials/mppi_cbf_reach_avoid.py` | User-facing breakage | Add deprecation shims or migration map |

## PR Sequence

### PR-0: Baseline + Gate Harness

Goal:
- Freeze metrics and define pass/fail guardrails before semantic reconciliation.

Scope:
- Benchmark command wrappers and artifact upload wiring.

Commands:
- `pytest -m "not slow" tests`
- `pytest -m "slow" tests`
- Key benchmark scripts in `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/tests/benchmarks/`.

Acceptance:
- Baseline timing and pass/fail artifacts published for later comparison.

Effort/Risk:
- 0.5d / Low.

### PR-1: Solver Status Contract Unification

Goal:
- Resolve SC-1/SC-2 with one canonical status policy and message layer.

Scope:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/optimization/quadratic_program/qp_solver_jaxopt.py`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/cbf_clf/cbf_clf_qp_generator.py`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/simulator.py`
- status-related tests in `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/tests/test_simulation/` and `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/tests/test_optimization/`.

Acceptance:
- Status decoding consistency for `-2,-1,0,1,2,3,4,5`.
- No mixed messaging around accepted `MAX_ITER` solutions.

Effort/Risk:
- 0.5-1.0d / High.

### PR-2: Public API Compatibility Layer

Goal:
- Resolve SC-3/SC-4 without forcing abrupt downstream migration.

Scope:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/utils/user_types.py`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/certificates/packager.py`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/cbf_clf/risk_aware_cbf_clf_qp_control_laws.py`
- docs updates for migration notes.

Acceptance:
- Legacy usage patterns exercised in compatibility tests.
- Migration section added to README/docs for changed signatures.

Effort/Risk:
- 1.0-2.0d / High.

### PR-3: Dependency Profile Hardening

Goal:
- Resolve SC-5/SC-6 and remove install-profile ambiguity.

Scope:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/pyproject.toml`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/.github/workflows/ci.yml`
- dependency-contract smoke tests for casadi/vis/codegen paths.

Acceptance:
- CI matrix proves: base + `test` + `codegen` + `vis` + `casadi` install/run contracts.

Effort/Risk:
- 1.5-2.5d / High.

### PR-4: Simulation Stack De-duplication

Goal:
- Resolve SC-7 by eliminating duplicated integrator policy logic.

Scope:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/backend.py`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/simulator_jit.py`
- parity tests for Python vs JIT outputs.

Acceptance:
- Shared integration helper used by both paths.
- Parity tests pass on representative systems.

Effort/Risk:
- 2.0-3.0d / High.

### PR-5: MPPI Equivalence and Performance Gates

Goal:
- Resolve SC-8 with measured confidence.

Scope:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/mppi/mppi_source.py`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/mppi/mppi_generator.py`
- deterministic behavior + runtime benchmark tests.

Acceptance:
- No behavioral regressions on fixed-seed scenarios.
- Runtime within agreed threshold vs PR-0 baseline.

Effort/Risk:
- 1.5-2.5d / High.

### PR-6: UX/Docs and Tutorial Migration

Goal:
- Resolve SC-9/SC-10 and reduce onboarding breakage.

Scope:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/callbacks.py`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/ui.py`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/README.md`
- tutorial deprecation shim files under `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/tutorials/`.

Acceptance:
- Non-TTY CI logs are readable and non-corrupt.
- Removed tutorial entrypoints have explicit migration pointer or shim.

Effort/Risk:
- 1.0-1.5d / Medium.

### PR-7: Structural Debt Cleanup

Goal:
- Remove process artifacts from runtime surface and reduce history noise impact.

Scope:
- policy decision on `.jules` tracking files and generated assets.

Acceptance:
- clear repository policy and cleaned tree.

Effort/Risk:
- 0.5-1.0d / Medium.

## Rollback Plan

1. Merge each PR as an isolated commit group.
2. If regression appears, revert latest PR merge first.
3. For critical safety regressions, prioritize reverting PR-1/PR-4/PR-5 before doc/tooling PRs.
4. Keep baseline artifacts from PR-0 to verify rollback recovery.

## Final Go/No-Go Gate

Go if all are true:
1. All PR acceptance criteria are green.
2. Dependency-profile CI matrix is fully green.
3. No benchmark regressions beyond threshold.
4. Simulation parity tests pass across Python and JIT paths.
5. Migration notes and tutorial mapping are published.

Else: No-Go.

## Machine-Readable Evidence Artifacts

1. Full commit inventory with category labels:
   - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_commit_inventory.tsv`
   - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_commit_inventory_local_bolt.tsv`
2. Dependency graph delta (internal module coupling):
   - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_dependency_graph_analysis.tsv`
   - global delta: module count `214 -> 218` (+4), edge count `257 -> 271` (+14)
3. Semantic conflict register (execution backlog):
   - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_semantic_conflict_matrix.tsv`
   - current tally: 18 conflicts (`8 High`, `10 Medium`)
   - estimated reconciliation effort from matrix: `11.5 - 20.5` engineer-days
4. Dependency-aware implementation backlog:
   - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_integration_backlog.tsv`
5. Merge readiness scorecard (resolved/partial/open by conflict ID):
   - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_merge_readiness_scorecard.md`
6. Remaining integration runbook (command-level):
   - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_remaining_integration_runbook.md`

## Tracking Rule

Use semantic conflict IDs (`SC-*`) as acceptance checklist items on each PR.
A PR touching a conflict cluster is not complete until the corresponding `SC-*` entry is either:
1. Resolved with tests, or
2. Deferred with owner/date and explicit risk acceptance note.

## Execution Status (Current Worktree)

Status date: 2026-02-17

Validated commands (all passing):
1. `bash scripts/run_premerge_gate.sh`
2. `bash scripts/run_premerge_gate.sh --with-dependency-contract`
3. `bash scripts/run_premerge_gate.sh --with-slow`
4. `bash scripts/run_premerge_gate.sh --with-dependency-contract --with-slow`
5. `pytest -m "not slow" tests`

Interpretation:
- `PR-0` baseline/gate harness is executable and green.
- Core parts of `PR-1` (solver status contract alignment) are implemented and validated in this worktree.
- `PR-2` is implemented (controller/signature adapter + SimulationResults compatibility surface + explicit manual-loop migration guidance).
- `PR-3` is partially implemented (`extra-profile-smoke` CI matrix wiring for `test/codegen/vis/casadi` profiles).
- `PR-4` is implemented (shared integration helper extraction + Python/JIT parity coverage).
- `PR-5` is implemented (deterministic MPPI parity + runtime regression guard tests).
- `PR-6` is implemented for migration/observability closure (tutorial alias shims + migration map + non-TTY output tests).
- `PR-7` is largely implemented (direct LQR tests + ROS/ROS2 wrapper smoke tests + quadrotor geometric tuple-contract fix/tests + optional CasADi smoke + `.jules` policy enforcement and generated artifact cleanup).
- Remaining ordered work spans residual `PR-3` and CI evidence closure for `PR-7`.
