"""lyapunov_fcn_catalog.py.

This file contains a catalog of candidate lyapunov functions and their associated gradients,
Hessians, etc., for use in control Lyapunov function-based controllers.
"""

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit

from cbfkit.utils.user_types import (
    LyapunovTuple,
)

# constants
N = 2  # number of states


###############################################################################
## Position Convergence
@jit
def V_pos(state: Array, goal: Array, r: float) -> Array:
    """Position goal function (drive Van der Pol oscillator to goal set around the origin).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg]
        r (float): goal set radius

    Returns
    -------
        ret (float): value of goal function evaluated at time and state
    """
    x, y, _t = state

    V = 0.5 * ((x - goal[0]) ** 2 + (y - goal[1]) ** 2 - r**2)

    return V


@jit
def dV_pos_dx(state: Array, goal: Array, r: float) -> Array:
    """Jacobian for position goal function (drive Van der Pol oscillator to goal set around the
    origin).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg]
        r (float): goal set radius

    Returns
    -------
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(V_pos)(state, goal, r)


@jit
def dV2_pos_dx2(state: Array, goal: Array, r: float) -> Array:
    """Hessian for position goal function (drive Van der Pol oscillator to goal set around the
    origin).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg]
        r (float): goal set radius

    Returns
    -------
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(V_pos))(state, goal, r)


def position(goal: Array, r: float) -> LyapunovTuple:
    """Callable that generates Lyapunov function and its associated.

    Args:
        goal (Array): goal position in inertial frame
        T (float): lookahead time horizon

    Returns
    -------
        LyapunovTuple: _description_
    """

    def v_func(t, x):
        return V_pos(jnp.hstack([x, t]), goal, r)  # type: ignore[return-value]

    def j_func(t, x):
        return dV_pos_dx(jnp.hstack([x, t]), goal, r)[:N]  # type: ignore[return-value]

    def h_func(t, x):
        return dV2_pos_dx2(jnp.hstack([x, t]), goal, r)[:N, :N]  # type: ignore[return-value]

    def p_func(t, x):
        return dV_pos_dx(jnp.hstack([x, t]), goal, r)[-1]  # type: ignore[return-value]

    def c_func(v):
        return 0.0

    return (
        v_func,
        j_func,
        h_func,
        p_func,
        c_func,
    )
