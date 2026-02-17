# Bolt -> Develop Merge Playbook

This playbook defines the execution order and gates for integrating `bolt` into `develop`
without stability, performance, or DX regressions.

## Preconditions

1. CI runs on `develop` and PRs targeting `develop`.
2. Base install contract is green (no extras required for core runtime).
3. Optional dependency error contracts are green (`casadi`, `vis` guidance).
4. Tutorial and example smoke tests do not include placeholder scripts.
5. Canonical rectifier implementation is single-source (`cbfkit.certificates.rectifiers`).

## Integration Sequence

1. Sync branch tips and record merge base:
   - `git fetch origin`
   - `git merge-base origin/develop origin/bolt`
2. Merge `origin/bolt` into a staging branch off `origin/develop` (no direct merge to `develop`).
3. Run full CI, then run local targeted checks:
   - solver failure-path tests
   - simulation status/reporting tests
   - example/tutorial slow tests
4. If green, open PR from staging branch into `develop` with required approvals.

### Recommended Local Command

Run the merge gate suite from repo root:

```bash
bash scripts/run_premerge_gate.sh --with-dependency-contract --with-slow
```

If your local environment does not include optional dependencies or you only want
the fast gate, run:

```bash
bash scripts/run_premerge_gate.sh
```

## Validation Gates

1. Functional:
   - `pytest -m "not slow" tests`
   - `pytest -m "slow" tests` (Python 3.10 lane)
2. Dependency contract:
   - `pip install -e .` succeeds
   - optional dep guidance assertions pass when optional deps are absent
3. Behavioral:
   - non-`SOLVED` QP statuses are treated as controller failures
   - solver status telemetry and simulator messaging remain consistent
4. Quality:
   - no executable test entry points are comment/docstring placeholders

## Rollback Strategy

1. Merge with a single merge commit (no squash) to keep rollback atomic.
2. If post-merge issues are detected:
   - `git revert -m 1 <merge-commit-sha>`
3. Keep pre-merge tag on `develop` for emergency pinning in CI/CD.

## Risk Labels

- High: Solver semantics, dependency/runtime contract changes.
- Medium: Simulation logging/reporting and tutorial execution harness.
- Low: Branch topology/textual merge conflicts when `develop` remains ancestor of `bolt`.
