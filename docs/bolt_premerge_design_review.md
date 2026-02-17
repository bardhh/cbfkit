# Bolt -> Develop Pre-Merge Design Review

## Scope and Evidence

Primary evidence used:
- `git log develop..bolt --stat` (requested local branch evidence)
- `git diff develop...bolt` (requested local branch evidence)
- `git log develop..origin/bolt --stat` and `git diff develop...origin/bolt` (tip-of-branch evidence; local `bolt` is behind by 16 commits)
- dependency graph/import delta analysis over `src/cbfkit`

Concrete branch topology:
- merge-base: `72b4d150fb9be9f18585dd09b2984cad64300036`
- requested local branch divergence: `git rev-list --left-right --count develop...bolt` => `0 215`
- divergence: `git rev-list --left-right --count develop...origin/bolt` => `0 231`
- merge shape: `develop` is an ancestor of `origin/bolt` (no Git-level textual conflicts expected)
- diff size: `227 files changed, 12378 insertions(+), 2875 deletions(-)`

High-churn hotspots from `git diff --numstat develop...origin/bolt`:
- `uv.lock` `+1026/-1217`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/cbf_clf/cbf_clf_qp_generator.py` `+410/-46`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/simulator.py` `+331/-138`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/mppi/mppi_source.py` `+103/-126`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/utils/user_types.py` `+177/-7`

Commit inventory:
- 215 local `bolt` commits are enumerated and categorized in `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_commit_inventory_local_bolt.tsv`.
- 231 `origin/bolt` tip commits are enumerated and categorized in `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_commit_inventory.tsv`.
- Category totals (heuristic + manual spot-check of high-churn commits):
  - `infra`: 85
  - `features`: 59
  - `bug fixes`: 45
  - `refactors`: 27
  - `experiments`: 15
- Automated merge commits: 56 (subset of infra).

## Executive Summary

Recommendation (as of 2026-02-17): **No-go for direct fast-forward to `develop`; Go for staged integration merge with enforced gates**.

Why:
1. Git merge conflicts are unlikely (`develop` -> `bolt` is fast-forwardable), but there are **semantic conflicts and policy drifts**.
2. Runtime behavior and API contracts shifted substantially in controller, solver, simulation, and packaging layers.
3. Dependency model moved from monolithic runtime deps to optional extras; migration gates are not yet fully explicit for all supported execution modes.
4. Structural debt increased (automation artifacts in repo history and broad coupling through `user_types`).

Effort to safe-merge (engineering + verification): **~10.5 to 14.5 engineer-days**.
Effort already completed in this worktree toward that total: **~4.5 to 5.5 engineer-days** (status/contract hardening + gate automation + compatibility + parity/perf regressions + simulation dedup + coverage hardening).
Current readiness/state tracker:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_merge_readiness_scorecard.md`

## Architectural and Dependency Divergence

### 1) Solver/controller/simulator contract drift
Evidence:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/cbf_clf/cbf_clf_qp_generator.py`
  - solver success policy is strict (`status == 1`), with explicit failure handling and NaN fallback.
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/optimization/quadratic_program/qp_solver_jaxopt.py`
  - introduces `solve_with_details`, warm-start state, and status remapping (`5` for max-iter-unsolved).
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/simulator.py`
  - status interpretation and warning strings include assumptions that can drift from controller policy.

Risk:
- Silent mismatch in safety expectations and diagnostics across layers.

### 2) Public API/typing shifts
Evidence:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/utils/user_types.py`
  - `ControllerCallable` now requires non-optional control input.
  - `SimulationResults` named tuple introduced; execution path now returns this type.
  - new typed dicts (`CbfClfQpConfig`, `CbfClfQpData`) used across constraints/controller layers.
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/certificates/packager.py`
  - `certificate_package` signature expanded (`input_style`, `use_factory`), and new `generate_certificate` API.

Risk:
- Downstream integrations with strict typing/signature assumptions may break at runtime or static-check time.

### 3) Dependency model and runtime optionality changed
Evidence:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/pyproject.toml`
  - runtime deps reduced and split into extras (`casadi`, `vis`, `codegen`, `test`, `dev`).
  - `control`, `pandas`, `tqdm`, `cvxpy`, `cvxpylayers` removed from runtime path.
  - `scipy` and `rich` added runtime.
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/optimization/quadratic_program/qp_solver_casadi.py`
  - casadi import is optional with explicit ImportError guidance.
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/utils/visualization.py`
  - matplotlib optional with explicit ImportError guidance.

Risk:
- Installation profile mismatch between environments can produce mode-specific failures (examples, visualization, codegen, casadi-backed flows).

### 4) Simulation stack refactor and duplicated integration logic
Evidence:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/backend.py`
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/simulator_jit.py`
- Shared helper now extracted:
  - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/integration_utils.py`
  - both backends call `integrate_with_cached_dynamics(...)` for forward-euler/RK4/custom integration.

Risk:
- Reduced behavioral drift risk between Python and JIT simulation paths.
- Remaining risk primarily in non-integration control-flow paths.

### 5) UI/observability infrastructure pivot
Evidence:
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/callbacks.py`
  - progress backend switched from `tqdm` to `rich`.
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/simulation/ui.py`
  - new output/progress utility layer.
- `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/utils/logger.py`
  - pandas removed; CSV writing now manual with list/dict modes.

Risk:
- Console/output behavior changes in CI/non-TTY contexts and logging consumers.

### 6) Structural debt introduced by branch process
Evidence:
- 56 `automated merge` commits in `develop..origin/bolt` history.
- `.jules` workflow artifacts were present in branch history; policy hardening now landed:
  - removed generated files: `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/.jules/relay_tasks.json`, `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/.jules/relay_planner.py`
  - enforcement: `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/scripts/check_jules_policy.sh`
  - policy doc: `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/jules_artifact_policy.md`

Risk:
- Review signal-to-noise degradation and future archaeology cost.

## Breaking Changes and Migration Requirements

### Breaking / behavior-changing surfaces
- Solver success semantics tightened in controller path (`status == 1` required).
- Failure fallback in CBF-CLF QP path moves toward explicit NaN over silent nominal fallback.
- `SimulationResults` return type can alter strict tuple-based assumptions despite tuple compatibility.
- Tutorial/example path churn and script renames/deletions:
  - removed: `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/tutorials/mppi_cbf_reach_avoid.py`
  - removed: `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/tutorials/mppi_stl_reach_avoid.py`
  - new canonical paths under `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/tutorials/` and `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/tutorials/tutorials/mppi_cbf_si/`

### Migration checklist (pre-merge)
1. Publish dependency profile matrix (base, test, codegen, vis, casadi) and make it executable in CI.
2. Add compatibility notes for updated solver statuses and controller failure semantics.
3. Provide tutorial backward-compatibility shims or clear deprecation notices for removed script entrypoints.
4. Decide policy on `.jules` artifacts (keep as tooling or move out of main package tree).

## Test Coverage and Regression Risk Assessment

Strengths:
- Very large test expansion (`91 test files changed`, `+6427/-51` lines), including dedicated safety/regression suites in controllers/simulation/certificates/optimization.

Gaps still visible:
- Remaining low/partial direct coverage for changed modules:
  - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/optimization/quadratic_program/qp_solver_casadi.py` (positive path depends on CasADi-enabled CI lane)
- Closed in current worktree:
  - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/ros/controller_wrappers.py` and `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/ros2/controller_wrappers.py` via `tests/test_ros_controller_wrappers_smoke.py`
  - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/utils/lqr.py` via `tests/test_utils/test_lqr.py`
  - `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/systems/quadrotor_6dof/controllers/geometric.py` via tuple-contract hardening and `tests/test_quadrotor_geometric_controller.py`

Performance regression risk areas (needs measured gate, not assumption):
- QP/controller hot path rewrites and normalization flow.
- JIT simulation output/log handling and progress hooks.
- MPPI rollout/vectorization rewrites in `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/src/cbfkit/controllers/mppi/mppi_source.py`.

## Detailed Conflict Matrix

| ID | Conflict Cluster | Concrete Evidence | Git Conflict? | Semantic Risk | Effort | Risk |
|---|---|---|---|---|---|---|
| C1 | Solver status policy alignment | `cbf_clf_qp_generator.py`, `qp_solver_jaxopt.py`, `simulation/simulator.py`, `tests/test_optimization/test_infeasible_qp.py` | No | High | 0.5-1.0d | High |
| C2 | Public API/type contract changes | `utils/user_types.py`, `certificates/packager.py`, `risk_aware_cbf_clf_qp_control_laws.py` | No | High | 1.0-2.0d | High |
| C3 | Dependency/extras migration | `pyproject.toml`, `uv.lock`, `qp_solver_casadi.py`, `utils/visualization.py`, `codegen/create_new_system/generate_model.py` | No | High | 1.5-2.5d | High |
| C4 | Simulation path divergence (Python vs JIT) | `simulation/backend.py`, `simulation/simulator_jit.py`, `simulation/simulator.py` | No | Medium-High | 2.0-3.0d | High |
| C5 | UI/logging stack swap | `simulation/callbacks.py`, `simulation/ui.py`, `utils/logger.py` | No | Medium | 0.5-1.0d | Medium |
| C6 | MPPI algorithm rewrite | `controllers/mppi/mppi_source.py`, `controllers/mppi/mppi_generator.py`, related tests | No | Medium-High | 1.5-2.5d | High |
| C7 | Tutorial/example entrypoint churn | `README.md`, `tutorials/*`, `tests/test_examples_and_tutorials.py` | No | Medium | 1.0-1.5d | Medium |
| C8 | Structural debt (automation artifacts/history noise) | `.jules/*`, 56 automated merge commits in inventory | No | Medium | 0.5-1.0d | Medium |
| C9 | Untested/low-direct-test changed modules | casadi solver positive-path lane | No | Medium | 0.5-1.0d | Medium |

Estimated total effort: **10.5 to 14.5 engineer-days**.

## Ordered Integration Plan (Execution-Ready)

### Phase 0: Freeze and Baseline (0.5d)
1. Create integration branch from `develop` (no direct fast-forward to protected branch).
2. Persist baseline metrics:
   - wall time of key simulation tests,
   - representative MPPI scenario runtimes,
   - core solver behavior snapshots.
3. Attach `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_commit_inventory.tsv` to PR for auditability.

### Phase 1: Blocking semantic fixes (C1 + C2 + C3) (3.0-5.5d)
1. Unify solver status semantics and messages across controller/solver/simulator.
2. Add compatibility shims and explicit migration notes for changed APIs.
3. Enforce dependency-contract checks for optional extras in CI:
   - base install,
   - `.[test]`,
   - `.[test,codegen]`,
   - `.[vis]`,
   - `.[casadi]` (smoke-level).

### Phase 2: Simulation/MPPI hardening (C4 + C6) (3.5-5.5d)
1. De-duplicate integration logic between Python and JIT simulator paths (single implementation boundary).
2. Add parity tests asserting Python vs JIT behavioral equivalence on selected systems.
3. Add benchmark thresholds for at least one controller loop and one MPPI rollout path.

### Phase 3: DevEx/doc hygiene (C5 + C7 + C8 + C9) (3.0-3.5d)
1. Verify rich/tqdm migration behavior in CI and non-TTY logs.
2. Add deprecation stubs or explicit replacements for removed tutorial entrypoints.
3. Decide and execute policy for `.jules` artifacts.
4. Close remaining coverage gap: green CasADi-enabled CI positive-path lane.

### Phase 4: Merge and Soak (1.0d)
1. Merge to `develop` after all validation gates pass.
2. 24-hour soak with nightly/full test + benchmark run.

## Rollback Strategy

1. Merge through a dedicated integration branch only.
2. Keep phase boundaries as separate merge commits to enable targeted rollback:
   - rollback semantic layer (C1-C3) without reverting docs/tooling,
   - rollback simulation/MPPI layer (C4/C6) independently.
3. If post-merge critical regression appears:
   - immediate revert of the latest phase merge commit,
   - preserve logs/artifacts and open hotfix branch from reverted `develop`.

## Post-Merge Validation Checklist

1. `pytest -m "not slow" tests` passes on Python 3.10/3.11/3.12.
2. Slow/simulation suites pass on 3.10 with deterministic seeds.
3. Optional dependency smoke checks pass for `casadi`, `vis`, `codegen` extras.
4. Core examples and tutorials referenced in README execute or fail with explicit, actionable guidance.
5. Solver status reporting is consistent across controller outputs, simulator warnings, and test expectations.
6. Python and JIT simulation paths produce equivalent trajectories within tolerance on canonical systems.
7. Benchmark deltas remain within accepted thresholds (document threshold per benchmark).
8. No uncontrolled repo artifact growth from tooling files (`.jules`, generated intermediates).

## Go/No-Go Decision

Current decision (2026-02-17):
- **Direct fast-forward merge to `develop`: No-Go**
- **Staged merge via integration branch and gates in `/Users/bardhh/.codex/worktrees/2948/cbfkit-remote/docs/bolt_develop_merge_playbook.md`: Go**

Gate evidence from current worktree:
1. `pytest -m "not slow" tests` => `180 passed, 1 skipped, 20 deselected`.
2. `bash scripts/run_premerge_gate.sh --with-dependency-contract --with-slow` => passed (includes slow tutorial/example lane).
3. Status/semantics hardening and tests are in place for:
   - solver status mapping and failure policy alignment,
   - optional dependency contract checks,
   - tutorial execution harness compatibility,
   - ROS/ROS2 controller wrapper fallback + metadata smoke coverage,
   - shared Python/JIT integration helper extraction for forward-euler/RK4/custom paths,
   - Python/JIT backend parity coverage for `forward_euler` and `runge_kutta_4`,
   - MPPI deterministic parity and coarse runtime-regression guardrails.

Remaining Go criteria for final merge to `develop`:
- dependency profile matrix green in CI across required Python versions,
- no unexplained benchmark regression against pre-merge baseline,
- all High-risk conflict clusters either resolved or explicitly accepted with owner/date.

## Quantified Risk Model

Scoring model:
- Probability (P): 1-5
- Impact (I): 1-5
- Risk Score = P * I
- Integration Cost: engineer-days

| Cluster | P | I | Score | Cost (days) | Priority |
|---|---:|---:|---:|---:|---|
| C1 Solver status policy alignment | 4 | 5 | 20 | 0.5-1.0 | P0 |
| C2 API/type contract changes | 4 | 4 | 16 | 1.0-2.0 | P0 |
| C3 Dependency/extras migration | 4 | 4 | 16 | 1.5-2.5 | P0 |
| C4 Simulation path divergence | 3 | 5 | 15 | 2.0-3.0 | P0 |
| C6 MPPI algorithm rewrite | 3 | 5 | 15 | 1.5-2.5 | P0 |
| C5 UI/logging stack swap | 3 | 3 | 9 | 0.5-1.0 | P1 |
| C7 Tutorial/example churn | 3 | 3 | 9 | 1.0-1.5 | P1 |
| C9 Low-direct-test modules | 2 | 3 | 6 | 0.5-1.0 | P2 |
| C8 Structural debt cleanup | 2 | 3 | 6 | 0.5-1.0 | P2 |

Interpretation:
- Any unresolved cluster with score >= 15 is a merge blocker.
- Score 8-14 requires explicit mitigation or temporary exception signed off in PR notes.
- Score <= 7 can be deferred post-merge with owner/date.
