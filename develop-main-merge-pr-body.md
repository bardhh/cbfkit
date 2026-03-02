# Develop -> Main Merge

## Summary
This PR merges `develop` (`7946095`) into `main` (`51c7c49`) after resolving the blocking infra mismatch in GPU devcontainer configuration.

## Diff Summary
- `527 files changed, 38243 insertions(+), 186478 deletions(-)`
- Scope counts:
  - `src`: 244
  - `tests`: 127
  - `examples`: 101
  - `tutorials`: 29
  - `infra`: 26

## Intentional Breaking Changes
No backward compatibility guarantee for this merge.

1. `simulation.execute` expanded signature and structured return (`SimulationResults`) in `src/cbfkit/simulation/simulator.py`.
2. Controller namespace migration from `controllers.model_based.cbf_clf_controllers` to `controllers.cbf_clf`.
3. Packaging contract updates in `pyproject.toml` (dynamic version, extras model, Python range).
4. Global JAX precision behavior set in `src/cbfkit/__init__.py` (`jax_enable_x64=True`).
5. New CLI entrypoint `cbfkit-bench`.

## Blocker Resolution: GPU Devcontainer
Resolved mismatch where GPU devcontainer referenced `cbfkit_gpu` but compose did not define it.

### Changes
- Added `cbfkit_gpu` service to `.devcontainer/docker-compose.yml`.
- Added `profiles: ["gpu"]` to `cbfkit_gpu`.
- Pointed `cbfkit_gpu` build to `gpu.Dockerfile`.
- Kept default CPU service (`cbfkit`) unchanged.

### Validation
- `docker compose -f .devcontainer/docker-compose.yml config` -> PASS
- `docker compose -f .devcontainer/docker-compose.yml --profile gpu config` -> PASS

## Verification Evidence (merge-prep branch)
- `pytest -m "not slow" tests` -> PASS (`232 passed, 4 skipped, 21 deselected`)
- `pytest -m "slow" tests` -> PASS (`21 passed, 236 deselected`)
- `ruff check src` -> PASS (`All checks passed!`)
- `mypy src/cbfkit` -> NON-BLOCKING WARNINGS (`50 errors in 25 files`; reported only)
- `git merge-base --is-ancestor main develop && echo OK` -> PASS (`OK`)

## Notes
- Branch: `bolt-dev/develop-main-merge-prep`
- This PR is ready for CI and review under the current merge policy.
