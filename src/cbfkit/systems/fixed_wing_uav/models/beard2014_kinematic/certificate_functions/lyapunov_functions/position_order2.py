"""
lyapunov_fcn_catalog.py

This file contains a catalog of candidate lyapunov functions and their associated
gradients, Hessians, etc., for use in control Lyapunov function-based controllers.
"""

import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
)

# constants
N = 6  # number of states


###############################################################################
## 2nd Order Goal Position
@jit
def clf(state: Array, goal: Array, c1: float, c2: float, e1: float, e2: float) -> Array:
    """Second order position goal function (drive fixed-wing UAV
    to a set of states within which a constant Vdot will drive it to
    the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        c1 (float): first-term gain on FxTS conditions
        c2 (float): second-term gain on FxTS conditions
        e1 (float): first-term exponent on FxTS conditions
        e2 (float): second-term exponent on FxTS conditions

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    x, y, z, v, psi, gamma, _t = state
    xg, yg, zg = goal

    xdot = v * jnp.cos(psi) * jnp.cos(gamma)
    ydot = v * jnp.cos(psi) * jnp.sin(gamma)
    zdot = v * jnp.sin(gamma)

    V = 0.5 * ((x - xg) ** 2 + (y - yg) ** 2 + (z - zg) ** 2)
    Vdot = (x - xg) * xdot + (y - yg) * ydot + (z - zg) * zdot

    return Vdot + c1 * V**e1 + c2 * V**e2


@jit
def clf_grad(state: Array, goal: Array, c1: float, c2: float, e1: float, e2: float) -> Array:
    """Jacobian for second order position goal function (drive fixed-wing UAV
    to a set of states within which a constant Vdot will drive it to
    the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        c1 (float): first-term gain on FxTS conditions
        c2 (float): second-term gain on FxTS conditions
        e1 (float): first-term exponent on FxTS conditions
        e2 (float): second-term exponent on FxTS conditions

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(clf)(state, goal, c1, c2, e1, e2)


@jit
def clf_hess(state: Array, goal: Array, c1: float, c2: float, e1: float, e2: float) -> Array:
    """Hessian for second order position goal function (drive fixed-wing UAV
    to a set of states within which a constant Vdot will drive it to
    the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        c1 (float): first-term gain on FxTS conditions
        c2 (float): second-term gain on FxTS conditions
        e1 (float): first-term exponent on FxTS conditions
        e2 (float): second-term exponent on FxTS conditions

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(clf))(state, goal, c1, c2, e1, e2)


###############################################################################
# Future-Focused CBF (Constant Velocity)
###############################################################################
position_order2 = certificate_package(clf, clf_grad, clf_hess, N)


# def position_order2(goal: Array, c1: float, c2: float, e1: float, e2: float) -> LyapunovTuple:
#     """Callable that generates Lyapunov function and its associated

#     Args:
#         goal (Array): goal position in inertial frame
#         T (float): lookahead time horizon

#     Returns:
#         LyapunovTuple: _description_
#     """
#     v_func: LyapunovCallable = lambda t, x: V_2pos(jnp.hstack([x, t]), goal, c1, c2, e1, e2)  # type: ignore[return-value]
#     j_func: LyapunovJacobianCallable = lambda t, x: dV_2pos_dx(jnp.hstack([x, t]), goal, c1, c2, e1, e2)[:N]  # type: ignore[return-value]
#     h_func: LyapunovHessianCallable = lambda t, x: dV2_2pos_dx2(jnp.hstack([x, t]), goal, c1, c2, e1, e2)[:N, :N]  # type: ignore[return-value]
#     p_func: LyapunovPartialCallable = lambda t, x: dV_2pos_dx(jnp.hstack([x, t]), goal, c1, c2, e1, e2)[-1]  # type: ignore[return-value]

#     return (
#         [v_func],
#         [j_func],
#         [h_func],
#         [p_func],
#     )
