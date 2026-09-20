# Contributing to CBFKit

Thanks for helping. This page covers the development setup, the checks CI runs, and how to
submit a change.

## Development setup

Requires Python 3.10–3.12. Clone the repository and install it editable with the development
extras:

```bash
git clone https://github.com/bardhh/cbfkit.git && cd cbfkit
pip install uv==0.9.12
uv sync --locked --extra dev
uv run --no-sync pre-commit install  # optional: run the hooks on every commit
```

`dev` excludes `manim`, `torch` and `mujoco`. Add them explicitly if your change touches those
integrations, e.g. `uv sync --locked --extra dev --extra mujoco` (the rendering test environment).

`pyproject.toml` declares supported library dependency ranges; `uv.lock` fixes the development
and regression-test environment. After changing dependencies, run `uv lock` and commit both
files. `uv sync --locked` rejects stale metadata. CI also tests an unlocked installation to
catch regressions within the supported ranges. Quality-tool versions are pinned in `quality`, which `dev` includes.

For focused work, use `--extra test` for pytest/coverage, `--extra test-all` for core and
example tests including optional solvers, `--extra quality` for lint/type tools, or
`--extra build` for distribution checks. MuJoCo tests need `--extra test --extra mujoco`;
on headless Linux install `libosmesa6 ffmpeg` and set `MUJOCO_GL=osmesa`. The quality job
also installs `all` and `mujoco` so their installed type information remains checked.

## Running the tests

```bash
uv run --no-sync pytest -m "not slow" tests --ignore=tests/test_mujoco
CBFKIT_TEST_MODE=1 uv run --no-sync pytest -m slow tests --ignore=tests/test_mujoco
CBFKIT_TEST_MODE=1 uv run --no-sync pytest tests/test_mujoco
uv run --no-sync pytest tests/test_controllers/test_cbf_clf_robustness.py   # a single file
```

Tests run on CPU: the root `conftest.py` sets `JAX_PLATFORM_NAME=cpu` unless it is already
set. `CBFKIT_TEST_MODE=1` also shortens simulations and skips plots in the examples and
tutorials, which is how `tests/test_examples_and_tutorials.py` smoke-runs them. Each smoke-test
subprocess has a 300-second timeout and reports captured output on failure. On macOS, if
native optional runtimes conflict when loaded together, run the affected test directories in
separate pytest processes and report that limitation.

## Linting, formatting and type checks

```bash
uv run --no-sync ruff check src/cbfkit
uv run --no-sync black --check src/cbfkit
uv run --no-sync isort --check-only src/cbfkit
uv run --no-sync mypy src/cbfkit
uv run --no-sync pre-commit run --all-files
```

All four source checks are blocking in CI and use the same scope in pre-commit. To apply
formatting, run `uv run --no-sync isort src/cbfkit` followed by
`uv run --no-sync black src/cbfkit`. Ruff checks unused imports (F401); declare intentional
re-exports with `__all__` or explicit aliases, and use a local suppression only for import
probes or registration side effects. E402 allows existing initialization order. Third-party
typing exceptions are listed in `mypy.ini`. Controller adapters, certificate packaging, and
simulation setup require typed definitions and check nested function bodies.

## What CI checks

The `CI` workflow (`.github/workflows/ci.yml`) runs on pushes to `main` or `develop` and on
pull requests targeting either branch:

- **quality**: Ruff, Black, isort and mypy, independently of runtime tests.
- **build** on Python 3.10, 3.11 and 3.12: core tests excluding the MuJoCo directory and
  slow tests, using locked `test-all`. Coverage and JUnit reports are uploaded.
- **examples** on Python 3.10: all slow tests outside the MuJoCo directory, including
  the example/tutorial smoke tests, using locked `test-all`.
- **mujoco** on Python 3.10: both fast and slow MuJoCo tests, using locked `test` and
  `mujoco` extras with OSMesa and FFmpeg. An encoder preflight fails if required tests
  skip. Tests needing large-memory G1 MJX compilation remain in the hardware lane.
- **core-import**: installs the package with no extras and imports `cbfkit`, so the core must
  stay importable without Gymnasium, PyTorch, MuJoCo or Manim.
- **jax-floor**: runs wrapper, solver-registry, certificate and eager/JIT/RNG parity tests
  against the oldest supported JAX (0.4.23) with NumPy < 2.
- **torch-bridge**: installs the optional `torch` extra on CPU (Python 3.11) and tests the
  PyTorch bridge, batch isolation and command mappings.
- **latest-dependencies**: installs unlocked `test-all` dependencies on Python 3.12 and runs the fast suite.
- **distribution**: builds and checks wheel/sdist metadata, installs each outside the checkout,
  and verifies package resources and generated dynamics. The wheel lane also steps the vendored
  cart-pole model offline.

The scheduled/manual `Optional hardware integration` workflow is enabled per job through
repository variables. Set `CBFKIT_MJX_RUNNER` to a trusted Linux runner label with at least
24 GiB RAM; it provisions and verifies pinned G1 assets before running offline policy tests.
Set `CBFKIT_CUDA_RUNNER` to a trusted runner whose `python` already provides CUDA JAX, CUDA
Torch, core CBFKit dependencies and pytest. These lanes fail when required tests skip.
Without these variables the jobs are visibly skipped; configuring them is an infrastructure
step, not evidence that the integrations have passed. Neither job runs on pull-request code.

## Public API compatibility and generated output

Use `cbfkit.certificates` for packaging/rectification, `cbfkit.utils.animators` for animation,
and `cbfkit.optimization.quadratic_program` for solver selection. Existing compatibility
imports remain supported. Solver results keep their `.primal`, `.status`, `.params` attributes
and tuple unpacking. The `showcase` and `reduced_order` modules remain public facades over
smaller private implementation modules.

Controller adapters retain automatic legacy signature detection. When a four-argument
callable has ambiguous parameter names, select its layout explicitly:
`setup_controller(fn, signature="key_data")` for `(t, x, key, data)` or
`signature="nominal_key"` for `(t, x, u_nom, key)`.

Write new generated models, trial output and local renders under ignored `results/`, or
use a temporary directory in tests. Keep reproducible scripts and measured benchmark
artifacts under version control when they support documented results. README showcase
media is intentional; avoid committing redundant render candidates. Local agent/browser
state and integration asset caches are ignored. Do not remove untracked research files as
part of routine cleanup.

Keep agent implementation plans, session logs, machine-specific run notes, and temporary
cleanup reports in ignored `.omc/` storage. Commit lasting usage and architecture guidance
alongside the relevant code or in contributor documentation. Recorded validation evidence
that supports published results belongs in the repository, with reproduction instructions.

## Performance baselines

Run on an otherwise idle machine, using the same locked Python environment for comparisons:

```bash
uv sync --locked --extra test
uv run --no-sync python benchmarks/baseline.py --output results/baseline.json
```

The runner repeats batches 1, 64 and 1024 three times in fresh CPU processes, disables the
persistent compilation cache, and records raw measurements plus median/min/max summaries.
It reports synchronized steady-state latency/throughput, warmup time, JAX backend compile
time during warmup (excluding tracing), and peak process host RSS (including imports, not
GPU VRAM). Missing compiler events or unsupported RSS measurement are reported as null.
The existing `batched_safety_filter.py --torch` mode remains available for bridge measurements.
Reports include revision, dirty state, Python/JAX versions and platform; keep local runs
under ignored `results/`. Do not compare absolute times across different hardware or enforce
noisy latency thresholds on shared CI runners. Compare safety failures and constraint
violations alongside speed; the baseline command fails if either count is nonzero.

## Submitting a change

1. Branch from `main` and keep the change focused. Separate refactors from behaviour changes.
2. Add or update tests under `tests/`. Examples that back a README tile should still run with
   `CBFKIT_TEST_MODE=1`.
3. If you add an optional dependency, put it behind an extra in `pyproject.toml`, raise an
   `ImportError` with the install hint when it is missing, and keep it out of the core import
   path.
4. If you change measured numbers that the README quotes (solver benchmarks, warehouse
   outcomes), regenerate them with the script in `benchmarks/` or `examples/` and say what
   hardware you used.
5. Open a pull request against `main`. Describe what changed and how you verified it.

Bug reports and feature requests go to the
[issue tracker](https://github.com/bardhh/cbfkit/issues).
