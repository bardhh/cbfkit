"""
position_ff.py

This file contains a catalog of candidate lyapunov functions and their associated
gradients, Hessians, etc., for use in control Lyapunov function-based controllers.
"""

import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
)

# from cbfkit.utils.user_types import (
#     CertificateCallable,
#     CertificateJacobianCallable,
#     CertificateHessianCallable,
#     CertificatePartialCallable,
#     CertificateTuple,
# )

# constants
N = 6  # number of states


###############################################################################
## Position Convergence
@jit
def clf(state: Array, goal: Array, T: float) -> Array:
    """Future-focused position goal function (drive fixed-wing UAV
    to a set of states within which a constant Vdot will drive it to
    the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        T (float): time horizon

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

    return V + Vdot * T


@jit
def clf_grad(state: Array, goal: Array, T: float) -> Array:
    """Jacobian for future-focused position goal function (drive fixed-wing UAV
    to a set of states within which a constant Vdot will drive it to
    the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        T (float): time horizon

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(clf)(state, goal, T)


@jit
def clf_hess(state: Array, goal: Array, T: float) -> Array:
    """Hessian for future-focused position goal function (drive fixed-wing UAV
    to a set of states within which a constant Vdot will drive it to
    the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        T (float): time horizon

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(clf))(state, goal, T)


###############################################################################
# Future-Focused CBF (Constant Velocity)
###############################################################################
position_ff = certificate_package(clf, clf_grad, clf_hess, N)


# def position_ff(goal: Array, T: float) -> LyapunovTuple:
#     """Callable that generates Lyapunov function and its associated

#     Args:
#         goal (Array): goal position in inertial frame
#         T (float): lookahead time horizon

#     Returns:
#         LyapunovTuple: _description_
#     """
#     v_func: LyapunovCallable = lambda t, x: V_posff(jnp.hstack([x, t]), goal, T)  # type: ignore[return-value]
#     j_func: LyapunovJacobianCallable = lambda t, x: dV_posff_dx(jnp.hstack([x, t]), goal, T)[:N]  # type: ignore[return-value]
#     h_func: LyapunovHessianCallable = lambda t, x: dV2_posff_dx2(jnp.hstack([x, t]), goal, T)[:N, :N]  # type: ignore[return-value]
#     p_func: LyapunovPartialCallable = lambda t, x: dV_posff_dx(jnp.hstack([x, t]), goal, T)[-1]  # type: ignore[return-value]

#     return (
#         [v_func],
#         [j_func],
#         [h_func],
#         [p_func],
#     )
