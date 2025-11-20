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
## Goal Velocity
def clf(goal: Array, r: float) -> Array:
    """Second order position goal function (drive fixed-wing UAV
    to a set of states within which a constant Vdot will drive it to
    the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal pos/vel vector [vg, yg, zg]
        r (float): acceptable radius (in m) around goal location

    Returns:
        ret (float): value of goal function evaluated at time and state

    """

    @jit
    def func(state_and_time: Array) -> Array:
        """ """
        _x, y, z, v, psi, gamma, _t = state_and_time
        vg, yg, zg = goal

        xdot = v * jnp.cos(psi) * jnp.cos(gamma)
        ydot = v * jnp.sin(psi) * jnp.cos(gamma)
        zdot = v * jnp.sin(gamma)

        xdot_d, ydot_d, zdot_d = vg, (yg - y), (zg - z)

        V = 0.05 * (((xdot - xdot_d)) ** 2 + ((ydot - ydot_d)) ** 2 + ((zdot - zdot_d)) ** 2) - r**2

        return V

    return func


def clf_grad(goal: Array, r: float) -> Array:
    """Jacobian for second order position goal function (drive fixed-wing UAV
    to a set of states within which a constant Vdot will drive it to
    the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        r (float): acceptable radius (in m) around goal location

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    jacobian = jacfwd(clf(goal, r))

    @jit
    def func(state_and_time: Array) -> Array:
        """ """
        return jacobian(state_and_time)

    return func


def clf_hess(goal: Array, r: float) -> Array:
    """Hessian for second order position goal function (drive fixed-wing UAV
    to a set of states within which a constant Vdot will drive it to
    the goal location within time T).

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal position vector [xg, yg, zg]
        r (float): acceptable radius (in m) around goal location

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    hessian = jacfwd(jacrev(clf(goal, r)))

    @jit
    def func(state_and_time: Array) -> Array:
        """ """
        return hessian(state_and_time)

    return func


###############################################################################
# Future-Focused CBF (Constant Velocity)
###############################################################################
velocity = certificate_package(clf, clf_grad, clf_hess, N)
