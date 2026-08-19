"""The 'fast' (PDIPM) solver falls back to a cold start when the warm start fails.

Regression for the G1 scramble + MPPI failure: at a replan boundary the carried
(s, lambda) from the previous step can be nearly degenerate; the warm-started solve
exhausts max_iter (status 2, NaN-prone) while a cold start converges well inside the
same budget. get_solver("fast") must return the cold solution in that case.
"""

import jax
import jax.numpy as jnp

from cbfkit.optimization.quadratic_program.qp_solver_pdipm import PdipmState, solve_qp_pdipm
from cbfkit.optimization.quadratic_program.solver_registry import get_solver

# min |x - (3, 3)|^2 s.t. x <= 1: solution (1, 1)
H = jnp.eye(2)
F = -2.0 * jnp.array([3.0, 3.0])
G = jnp.eye(2)
HV = jnp.ones(2)
BAD = PdipmState(  # degenerate corner far from the solution: s and lambda both ~ 0
    x=jnp.array([-50.0, 40.0]), s=jnp.full(2, 1e-12), dual=jnp.full(2, 1e-12), iter_num=0
)


def test_bad_warm_start_actually_fails_the_raw_solver():
    _, status, _ = solve_qp_pdipm(2.0 * H, F, G, HV, warm_start=BAD, max_iter=8)
    assert int(status) == 2  # the premise of the fallback
    _, status_cold, _ = solve_qp_pdipm(2.0 * H, F, G, HV, warm_start=None, max_iter=8)
    assert int(status_cold) == 1


def test_fast_solver_recovers_via_cold_restart():
    solver = get_solver("fast", max_iter=8)
    sol = solver(H, F, G, HV, None, None, init_params=(jnp.zeros(2), BAD))
    assert int(sol.status) == 1
    assert jnp.allclose(sol.primal, jnp.ones(2), atol=1e-5)


def test_fast_solver_cold_restart_is_jittable():
    solver = get_solver("fast", max_iter=8)

    @jax.jit
    def solve(f_vec, warm):
        return solver(H, f_vec, G, HV, None, None, init_params=(jnp.zeros(2), warm)).primal

    assert jnp.allclose(solve(F, BAD), jnp.ones(2), atol=1e-5)
