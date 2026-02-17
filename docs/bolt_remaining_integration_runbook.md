# Bolt -> Develop Remaining Integration Runbook

Status date: 2026-02-17  
Use this after current worktree hardening to complete remaining PRs (`PR-2` to `PR-7`).

Already completed in current worktree:
- PR-2 closure: legacy controller/signature compatibility covered for `sim.execute(...)` and `sim.simulator(...)`.
- PR-4 closure: shared integration helper extracted for both Python and JIT simulation backends.
- PR-5 closure: deterministic MPPI parity tests + runtime regression guard.
- PR-6 closure: tutorial migration map + compatibility shims + non-TTY observability tests.
- PR-7 structural policy closure: `.jules` artifact policy + enforcement + generated artifact cleanup.
- PR-7 coverage closure (local): direct LQR tests, ROS/ROS2 wrapper smoke tests, and quadrotor geometric tuple-contract tests.

## 0) Create Staging Branch

```bash
git fetch origin
git switch -c bolt-dev/bolt-merge-staging origin/develop
git merge --no-ff origin/bolt
```

If merge is not clean at this point, stop and re-run semantic triage against updated branch tips.

## 1) Baseline Validation (must stay green)

```bash
pytest -m "not slow" tests
bash scripts/run_premerge_gate.sh --with-dependency-contract --with-slow
```

## 2) PR-2 API Compatibility Layer (SC-4, SC-5)

Actions:
- Add compatibility adapter/tests for `ControllerCallable` call-site expectations.
- Add compatibility tests and migration notes for `SimulationResults` tuple-like consumers.

Exit criteria:
- legacy signatures/usages pass dedicated compatibility tests.

## 3) PR-3 Dependency Profile Matrix (SC-6, SC-7, SC-8)

Actions:
- Extend CI to include explicit install/smoke jobs for:
  - base
  - `.[test]`
  - `.[test,codegen]`
  - `.[vis]`
  - `.[casadi]`

Local smoke commands:

```bash
uv pip install -e ".[vis]"
python -c "import cbfkit.utils.visualization as v; v.require_visualization()"
```

```bash
uv pip install -e ".[casadi]"
python -c "from cbfkit.optimization.quadratic_program.qp_solver_casadi import solve; print(callable(solve))"
```

## 4) PR-4 Simulation De-duplication (SC-9, SC-10)

Actions:
- Extract shared integration helper used by both:
  - `src/cbfkit/simulation/backend.py`
  - `src/cbfkit/simulation/simulator_jit.py`
- Add parity tests (Python vs JIT) on canonical systems.

Exit criteria:
- parity assertions pass with tolerances defined in tests.

## 5) PR-5 MPPI Parity + Performance Gates (SC-11)

Actions:
- Add fixed-seed MPPI behavior parity scenarios.
- Add benchmark thresholds and compare against baseline artifacts.

Exit criteria:
- no unexplained behavior drift;
- benchmark deltas within documented threshold.

## 6) PR-6 UX and Tutorial Migration (SC-12, SC-13)

Actions:
- Add non-TTY log assertion in CI for rich progress output.
- Publish explicit migration map for renamed/removed tutorial entrypoints.
- Keep compatibility aliases where low-cost and high-usage.

Exit criteria:
- docs and tests cover all supported tutorial commands.

## 7) PR-7 Structural Debt + Coverage Gaps (SC-14, SC-15, SC-16, SC-17, SC-18)

Actions:
- Decide `.jules` retention policy and apply consistently.
- Add CasADi-enabled smoke test lane.
- Add direct tests for `src/cbfkit/utils/lqr.py`, ROS/ROS2 controller wrappers, and quadrotor geometric controller call-sites.

Exit criteria:
- no unresolved structural debt decision;
- coverage gap items tested (remaining blocker is green CasADi-enabled CI evidence).

## 8) Final Merge Gate

Required green checks:

```bash
pytest -m "not slow" tests
bash scripts/run_premerge_gate.sh --with-dependency-contract --with-slow
```

Plus:
- CI dependency profile matrix fully green.
- benchmark gate green with no unexplained regressions.

## 9) Rollback

If regression appears after merging staged PRs:

```bash
git revert -m 1 <merge_commit_sha>
```

Then:
- re-run baseline validation;
- open hotfix branch from reverted `develop`;
- keep failing artifacts for triage.
