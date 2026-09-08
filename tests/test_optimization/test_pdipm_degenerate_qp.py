"""PDIPM on a degenerate-optimal QP: best-iterate tracking and the non-finite-step latch.

The fixture ``data/scramble_degenerate_qp.npz`` is a real 42-variable / 124-row relaxable
CBF-QP captured from the G1 scramble (goal planner, seed 1, t = 13.2 s): the optimum sits
on the control bound with a degenerate active set. Measured behaviour of the raw Mehrotra
loop before the fix: residual 2.8e-6 at iteration 15, *rising* afterwards, NaN at 18 --
so max_iter = 16 returned a good solution with status 2 and max_iter >= 32 returned NaN.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.optimization.quadratic_program.qp_solver_pdipm import solve_qp_pdipm
from cbfkit.optimization.quadratic_program.solver_registry import get_solver

DATA = os.path.join(os.path.dirname(__file__), "data", "scramble_degenerate_qp.npz")


@pytest.fixture(scope="module")
def qp():
    z = np.load(DATA)
    return (jnp.asarray(z["H"]), jnp.asarray(z["f"]), jnp.asarray(z["G"]), jnp.asarray(z["h"]))


# The reference solution (cvxopt and the pre-breakdown PDIPM iterate agree).
X_REF = np.array([1.0, -0.5475])


@pytest.mark.parametrize("max_iter", [16, 32, 64, 128])
def test_pdipm_returns_the_best_iterate_never_nan(qp, max_iter):
    H, f, G, h = qp
    sol, status, state = solve_qp_pdipm(2.0 * H, f, G, h, max_iter=max_iter)
    assert bool(jnp.isfinite(sol).all()), f"non-finite solution at max_iter={max_iter}"
    assert np.allclose(np.asarray(sol[:2]), X_REF, atol=1e-3)
    assert bool(jnp.isfinite(state.x).all() & jnp.isfinite(state.s).all())


def test_fast_solver_with_loose_tol_reports_solved(qp):
    """The residual stalls at 2.8e-6 (scale set by the 1e3 slack penalty): tol=1e-5 accepts
    it -- and the freeze-on-converge guard then prevents the late-stage blow-up entirely."""
    H, f, G, h = qp
    solver = get_solver("fast", max_iter=32, tol=1e-5)
    sol = solver(H, f, G, h, None, None)
    assert int(sol.status) == 1
    assert np.allclose(np.asarray(sol.primal[:2]), X_REF, atol=1e-3)
