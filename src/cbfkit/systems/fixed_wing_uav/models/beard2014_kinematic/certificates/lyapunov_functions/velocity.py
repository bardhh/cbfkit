"""lyapunov_fcn_catalog.py.

This file contains a catalog of candidate Lyapunov functions and their associated
gradients, Hessians, etc., for use in control Lyapunov function-based controllers.
"""

from typing import Callable

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit

from cbfkit.certificates import certificate_package

# constants
N = 6  # number of states


###############################################################################
# Goal Velocity
###############################################################################
def clf(goal: Array, r: float) -> Callable[[Array], Array]:
    """Second order position goal function.

    Drives a fixed-wing UAV to states where a constant Vdot yields convergence to
    the goal within time T.

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal pos/vel vector [vg, yg, zg]
        r (float): acceptable radius around goal location
    """

    @jit
    def func(state_and_time: Array) -> Array:
        """Computes the Lyapunov function."""
        x, y, z, v, psi, gamma, t = state_and_time
        vg, yg, zg = goal

        xdot = v * jnp.cos(psi) * jnp.cos(gamma)
        ydot = v * jnp.sin(psi) * jnp.cos(gamma)
        zdot = v * jnp.sin(gamma)

        xdot_d = vg
        ydot_d = yg - y
        zdot_d = zg - z

        V = 0.05 * ((xdot - xdot_d) ** 2 + (ydot - ydot_d) ** 2 + (zdot - zdot_d) ** 2) - r**2

        return V

    return func


def clf_grad(goal: Array, r: float) -> Callable[[Array], Array]:
    """Jacobian of the Lyapunov function for velocity control."""
    jacobian = jacfwd(clf(goal, r))

    @jit
    def func(state_and_time: Array) -> Array:
        """Computes the Jacobian of the Lyapunov function."""
        return jacobian(state_and_time)

    return func


def clf_hess(goal: Array, r: float) -> Callable[[Array], Array]:
    """Hessian of the Lyapunov function for velocity control."""
    hessian = jacfwd(jacrev(clf(goal, r)))

    @jit
    def func(state_and_time: Array) -> Array:
        return hessian(state_and_time)

    return func


###############################################################################
# Future-Focused CBF (Constant Velocity)
###############################################################################
velocity = certificate_package(clf, clf_grad, clf_hess, N)
