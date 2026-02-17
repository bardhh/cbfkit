# Bolt -> Develop Merge Readiness Scorecard

Status date: 2026-02-17  
Scope: current worktree against `develop...origin/bolt`.

## Evidence Snapshot

- Topology: `git rev-list --left-right --count develop...origin/bolt` => `0 231`.
- Requested local branch check: `git rev-list --left-right --count develop...bolt` => `0 215` (local `bolt` is behind `origin/bolt` by 16 commits).
- Diff size: `git diff --shortstat develop...origin/bolt` => `227 files changed, 12378 insertions(+), 2875 deletions(-)`.
- Textual conflicts: `git merge-tree $(git merge-base develop origin/bolt) develop origin/bolt` => no conflict markers.
- Validation (all passing in current worktree):
  - `pytest -m "not slow" tests` => `180 passed, 1 skipped, 20 deselected`
  - `bash scripts/run_premerge_gate.sh --with-dependency-contract --with-slow`
- Remote evidence availability:
  - `git ls-remote --heads origin` confirms `refs/heads/bolt` at `b774d2010c458e0b2fb1268a538ca0be3cb902c8`.
  - `GET /repos/bardhh/cbfkit/pulls?state=open&base=develop&head=bardhh:bolt` returned no open PR.
  - `GET /repos/bardhh/cbfkit/actions/runs?branch=bolt` returned no workflow runs.
- Local CasADi-enabled install lane could not be reproduced in this workspace due environment policy constraints; CI evidence remains the source of truth for CasADi-enabled profile closure.

## Conflict Status Matrix

Legend:
- `Resolved`: acceptance intent satisfied in current worktree and covered by passing checks.
- `Partial`: mitigation landed, but at least one acceptance criterion remains.
- `Open`: no substantive mitigation landed yet.

Current tally: `Resolved=15`, `Partial=3`, `Open=0`.

| ID | Status | Evidence landed | Residual gap | Target PR |
|---|---|---|---|---|
| SC-1 | Resolved | strict success policy in `src/cbfkit/controllers/cbf_clf/cbf_clf_qp_generator.py`, simulator alignment in `src/cbfkit/simulation/simulator.py`, tests in `tests/test_simulation/test_solver_status_policy.py` | none | PR-1 |
| SC-2 | Resolved | `-2` mapped in `src/cbfkit/simulation/simulator.py`, error-format test in `tests/test_simulation/test_error_reporting.py` | none | PR-1 |
| SC-3 | Resolved | max-iter status warning semantics updated in `src/cbfkit/simulation/simulator.py`, policy test in `tests/test_simulation/test_solver_status_policy.py` | none | PR-1 |
| SC-4 | Resolved | adapter in `src/cbfkit/controllers/utils.py` (`setup_controller`) and automatic wiring in `src/cbfkit/simulation/simulator.py`; explicit manual-loop guidance in `README.md`; compatibility test in `tests/test_controllers/test_controller_utils_compatibility.py` | none | PR-2 |
| SC-5 | Resolved | legacy aliases and tuple helper in `src/cbfkit/utils/user_types.py`; compatibility tests in `tests/test_simulation/test_simulation_results_api.py` | none | PR-2 |
| SC-6 | Partial | CI has `dependency-contract` and `extra-profile-smoke` jobs in `.github/workflows/ci.yml` covering `test/codegen/vis/casadi` install+smoke contracts | await green CI evidence across all profile lanes | PR-3 |
| SC-7 | Partial | actionable missing-dependency contract validated in `scripts/validate_optional_dependency_contracts.py`; enabled CasADi profile smoke lane in `.github/workflows/ci.yml` | await green CI evidence for CasADi-enabled lane | PR-3/PR-7 |
| SC-8 | Resolved | visualization contract test in `tests/test_utils/test_visualization_optional_dependency.py` and contract script | none | PR-3 |
| SC-9 | Resolved | shared helper `src/cbfkit/simulation/integration_utils.py` now used by `src/cbfkit/simulation/backend.py`; parity coverage in `tests/test_simulation/test_backend_parity.py` | none | PR-4 |
| SC-10 | Resolved | shared helper `src/cbfkit/simulation/integration_utils.py` now used by `src/cbfkit/simulation/simulator_jit.py`; parity coverage in `tests/test_simulation/test_backend_parity.py` and `tests/test_simulation/test_simulator_jit_rk4.py` | none | PR-4 |
| SC-11 | Resolved | deterministic/parity+performance guard tests in `tests/test_controllers/test_mppi_parity_and_perf.py` and existing seed-stability checks in `tests/test_controllers/test_mppi_determinism.py` | none | PR-5 |
| SC-12 | Resolved | non-TTY plain-output assertions in `tests/test_simulation/test_ui.py` | none | PR-6 |
| SC-13 | Resolved | migration map in `docs/tutorial_entrypoint_migration.md`; compatibility shims in `tutorials/mppi_cbf_reach_avoid.py`, `tutorials/mppi_stl_reach_avoid.py`, `tutorials/single_integrator_dynamic_obstacles.py`; tests in `tests/test_tutorial_migration_aliases.py` | none | PR-6 |
| SC-14 | Resolved | policy document `docs/jules_artifact_policy.md`; enforced by `scripts/check_jules_policy.sh` + `scripts/run_premerge_gate.sh`; generated `.jules/relay_planner.py` and `.jules/relay_tasks.json` removed | none | PR-7 |
| SC-15 | Partial | base-path CasADi guidance checks plus optional positive smoke test in `tests/test_optimization/test_qp_solver_casadi_optional.py`; CI `casadi` profile smoke lane in `.github/workflows/ci.yml` | await green CI evidence for `casadi` profile lane | PR-7 |
| SC-16 | Resolved | direct LQR test in `tests/test_utils/test_lqr.py` for `compute_lqr_gain` reference system | none | PR-7 |
| SC-17 | Resolved | ROS and ROS2 wrapper smoke coverage in `tests/test_ros_controller_wrappers_smoke.py`; merge-gate inclusion via `scripts/run_premerge_gate.sh` | none | PR-7 |
| SC-18 | Resolved | tuple-compatible quadrotor dynamics handling in `src/cbfkit/systems/quadrotor_6dof/controllers/geometric.py`; direct coverage in `tests/test_quadrotor_geometric_controller.py` | none | PR-7 |

## Remaining Effort

Assumption for estimate rollup:
- `Open`: 100% of listed range remaining.
- `Partial`: 50% of listed range remaining.

Computed remaining reconciliation effort:
- Conflict work remaining: `1.25 - 2.25` engineer-days.
- With merge/soak execution item (`WI-11`): `2.25 - 3.25` engineer-days.

## Merge Recommendation

- Direct fast-forward merge into `develop`: `No-Go`.
- Staged integration branch merge with gated PR sequence: `Go` (subject to closing all High-risk `Open/Partial` items and benchmark gate).
