# Contributing to CBFKit

Thanks for helping. This page covers the development setup, the checks CI runs, and how to
submit a change.

## Development setup

Requires Python 3.10–3.12. Clone the repository and install it editable with the development
extras:

```bash
git clone https://github.com/bardhh/cbfkit.git && cd cbfkit
pip install -e ".[dev]"          # core + `all` extras + pytest, ruff, black, mypy, jupyter
pip install pre-commit isort     # not in `dev`; needed for the hooks below
pre-commit install               # optional: run the hooks on every commit
```

`dev` excludes `manim`, `torch` and `mujoco`. Add them explicitly if your change touches those
integrations, e.g. `pip install -e ".[dev,mujoco]"` (this is what CI installs on Python 3.10).

## Running the tests

```bash
pytest -m "not slow" tests                       # fast suite, what CI runs on every Python version
CBFKIT_TEST_MODE=1 pytest -m "slow" tests        # slow suite, CI runs it on Python 3.10 only
CBFKIT_TEST_MODE=1 pytest tests/test_mujoco      # MuJoCo/MJX tests, skipped when the extra is absent
pytest tests/test_controllers/test_cbf_clf_robustness.py   # a single file
```

Tests run on CPU: the root `conftest.py` sets `JAX_PLATFORM_NAME=cpu` unless it is already
set. `CBFKIT_TEST_MODE=1` also shortens simulations and skips plots in the examples and
tutorials, which is how `tests/test_examples_and_tutorials.py` smoke-runs them.

## Linting, formatting and type checks

```bash
ruff check src/cbfkit        # E and F rules; F401, E402 and E501 are ignored
black src/cbfkit             # line length 100
isort src/cbfkit             # black-compatible profile
mypy src/cbfkit              # see mypy.ini for exclusions
pre-commit run --all-files   # the above plus whitespace and YAML checks; its mypy hook skips
                             # all of examples/, a wider exclusion than mypy.ini
```

Note that CI runs `mypy src/cbfkit || true`, so a type error never fails CI. Pre-commit is the
only gate that enforces mypy; please run it before opening a pull request.

## What CI checks

The `CI` workflow (`.github/workflows/ci.yml`) runs on pushes to `main` or `develop` and on
pull requests targeting either branch:

- **build** on Python 3.10, 3.11 and 3.12: `pytest -m "not slow" tests`. On 3.10 it also runs
  ruff, mypy (non-blocking), the slow suite, and installs the `mujoco` extra.
- **core-import**: installs the package with no extras and imports `cbfkit`, so the core must
  stay importable without Gymnasium, PyTorch, MuJoCo or Manim.
- **jax-floor**: runs the safety-filter wrapper tests against the oldest supported JAX (0.4.23)
  with NumPy < 2.
- **torch-bridge**: installs the optional `torch` extra on CPU (Python 3.11) and tests the
  PyTorch bridge, batch isolation and command mappings.

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
