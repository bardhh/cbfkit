"""Regression tests for ``generate_mpc_solver_quadratic_cost_linear_dynamics``.

History: this solver had two latent cost-formulation bugs (the module had no tests
or examples exercising it):

1. The reference horizon was flattened state-major (``.T.flatten()``) while the
   Hessian block ``kron(I_N, Q)`` and the decision-vector unpack are time-major.
   That scrambled per-state-dimension weights against per-state-dimension references.

2. The linear cost term was ``-Q @ r`` instead of ``-2*Q @ r``. The jaxopt QP solver
   minimizes ``xᵀHx + fᵀx`` with no implicit 1/2, so a tracking cost
   ``(x-r)ᵀQ(x-r)`` produces ``f = -2Q r``. The missing factor of 2 made the QP
   optimum collapse to ``r/2``: a goal of (4, 4) parked the system at (2, 2).

These tests pin both corrections by driving a discrete-time double-integrator and
asserting the realized trajectory actually reaches the goal at the correct location.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.optimization.mpc.quadratic_cost_linear_dynamics import (
    generate_mpc_solver_quadratic_cost_linear_dynamics,
)


def _double_integrator_2d(dt: float = 0.1):
    A = jnp.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    B = jnp.array([[0.0, 0.0], [0.0, 0.0], [dt, 0.0], [0.0, dt]])
    return A, B


def _run_mpc(goal_xy, n_steps: int = 40, horizon: int = 20, dt: float = 0.1):
    A, B = _double_integrator_2d(dt)
    Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))
    R = 0.1 * jnp.eye(2)
    Qn = 50.0 * Q
    solve = generate_mpc_solver_quadratic_cost_linear_dynamics(A, B, Q, R, Qn, horizon)

    goal = jnp.array([goal_xy[0], goal_xy[1], 0.0, 0.0])
    ref_horizon = jnp.tile(goal, (horizon, 1))
    x = jnp.zeros(4)
    for _ in range(n_steps):
        concatenated_x_xr = jnp.vstack([x.reshape(1, -1), ref_horizon])
        _x_opt, u_opt = solve(concatenated_x_xr)
        x = A @ x + B @ u_opt[:, 0]
    return np.asarray(x)


def test_mpc_reaches_goal_no_factor_of_two_collapse():
    """Goal (4, 4) must produce x_T ≈ (4, 4) — pre-fix it parked at (2, 2)."""
    x_final = _run_mpc((4.0, 4.0))
    assert np.allclose(x_final[:2], [4.0, 4.0], atol=1e-2), (
        f"MPC final position {x_final[:2]} did not reach goal (4, 4). "
        "If it landed near (2, 2) the linear-cost factor-of-2 regressed."
    )
    # Terminal velocity should also be ~0 (system is fully regulated).
    assert np.allclose(
        x_final[2:], [0.0, 0.0], atol=1e-2
    ), f"Terminal velocity {x_final[2:]} non-zero; MPC failed to settle."


@pytest.mark.parametrize("goal_xy", [(3.0, 1.0), (1.0, 3.0), (-2.0, 2.5)])
def test_mpc_tracks_asymmetric_goal_no_axis_scrambling(goal_xy):
    """Asymmetric goals catch state-major vs time-major reference ordering.

    If the cost vector reverts to state-major (the old ``.T.flatten()``),
    px-weights pair with py-references and the realized x/y will swap or smear.
    Symmetric goals like (4, 4) cannot detect this; these asymmetric goals can.
    """
    x_final = _run_mpc(goal_xy, n_steps=60)
    assert np.allclose(x_final[:2], goal_xy, atol=2e-2), (
        f"MPC reached {x_final[:2]} for goal {goal_xy}. "
        "If x and y are swapped/smeared, the reference time/state ordering regressed."
    )
