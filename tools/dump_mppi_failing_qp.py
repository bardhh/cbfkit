"""Capture the first QP that solve_qp_fast would see inside mppi_cbf_reach_avoid.

Saves P, q, G, h to tests/test_optimization/fixtures/mppi_failing_qp.npz for
use as a regression fixture.
"""
import os
import sys

os.environ["CBFKIT_TEST_MODE"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MPLBACKEND"] = "Agg"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import jax
import numpy as np
from cbfkit.optimization.quadratic_program import qp_solver_fast as fast_mod

FIXTURE_DIR = os.path.join(ROOT, "tests/test_optimization/fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)
DUMP_PATH = os.path.join(FIXTURE_DIR, "mppi_failing_qp.npz")

_dumped = [False]


def _host_dump(P, q, G, h):
    if _dumped[0]:
        return
    _dumped[0] = True
    np.savez(
        DUMP_PATH,
        P=np.asarray(P),
        q=np.asarray(q),
        G=np.asarray(G),
        h=np.asarray(h),
    )
    print(f"[dump] saved fixture {DUMP_PATH}", file=sys.stderr)


_orig_solve = fast_mod.solve_qp_fast


def _instrumented(P, q, G, h, **kwargs):
    jax.debug.callback(_host_dump, P, q, G, h)
    return _orig_solve(P, q, G, h, **kwargs)


fast_mod.solve_qp_fast = _instrumented

# Force solver registry to use "fast" so our shim sees the call.
# Setting the env var is the same hook the integration test suite uses
# (tests/conftest.py reads CBFKIT_QP_SOLVER in Task 10).
os.environ["CBFKIT_QP_SOLVER"] = "fast"
from cbfkit.optimization.quadratic_program import solver_registry as _sr

_orig_get = _sr.get_solver


def _patched_get(name="jaxopt", **kwargs):
    return _orig_get("fast", **{k: v for k, v in kwargs.items() if k in ("max_iter", "tol")})


# Both patches are required: cbf_clf_qp_generator does
# `from ...solver_registry import get_solver` (a name binding into its own
# module), so rebinding the registry attribute alone would NOT redirect
# the generator's existing local name. We patch both module attributes.
_sr.get_solver = _patched_get
from cbfkit.controllers.cbf_clf import cbf_clf_qp_generator as _gen

_gen.get_solver = _patched_get

import runpy

try:
    runpy.run_path("tutorials/mppi_cbf_reach_avoid.py", run_name="__main__")
except SystemExit:
    pass
print("[done]", file=sys.stderr)
