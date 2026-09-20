"""position_ff.py.

This file contains a catalog of candidate lyapunov functions and their associated gradients,
Hessians, etc., for use in control Lyapunov function-based controllers.
"""

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit

from cbfkit.certificates import certificate_package

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
    """Future-focused position goal function (drive fixed-wing UAV to a set of states within which a
    constant Vdot will drive it to the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        T (float): time horizon

    Returns
    -------
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
    """Jacobian for future-focused position goal function (drive fixed-wing UAV to a set of states
    within which a constant Vdot will drive it to the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        T (float): time horizon

    Returns
    -------
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(clf)(state, goal, T)


@jit
def clf_hess(state: Array, goal: Array, T: float) -> Array:
    """Hessian for future-focused position goal function (drive fixed-wing UAV to a set of states
    within which a constant Vdot will drive it to the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        T (float): time horizon

    Returns
    -------
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(clf))(state, goal, T)


###############################################################################
# Future-Focused CBF (Constant Velocity)
###############################################################################
position_ff = certificate_package(clf, clf_grad, clf_hess, N)
