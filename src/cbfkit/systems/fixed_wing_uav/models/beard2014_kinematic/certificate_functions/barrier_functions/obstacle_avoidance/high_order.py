"""
"""

import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array
from typing import List
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
)

N = 6


###############################################################################
# 2nd Order CBF
###############################################################################


def cbf(obstacle: Array, ellipsoid: List[float], alpha: float) -> Array:
    """Obstacle avoidance constraint function for Fixed-Wing UAV. Super-level set convention.

    Args:
        obstacle (Array): x, y, z location of obstacle
        ellipsoid (List): list of 3D ellipsoid parameters
        alpha (float): class K function (for 2nd order CBF)

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    x_o, y_o, z_o = obstacle
    a1, a2, a3 = ellipsoid

    @jit
    def func(state_and_time: Array) -> Array:

        # States
        x_e, y_e, z_e, v_e, psi_e, gamma_e, _t = state_and_time
        # x_o, y_o, z_o, v_o, psi_o, gamma_o, _t = obstacle

        # Velocities
        xe_dot = v_e * jnp.cos(psi_e) * jnp.cos(gamma_e)
        ye_dot = v_e * jnp.sin(psi_e) * jnp.cos(gamma_e)
        ze_dot = v_e * jnp.sin(gamma_e)
        xo_dot = 0.0  # v_o * jnp.cos(psi_o) * jnp.cos(gamma_o)
        yo_dot = 0.0  # v_o * jnp.sin(psi_o) * jnp.cos(gamma_o)
        zo_dot = 0.0  # v_o * jnp.sin(gamma_o)

        # dpos, dvel
        dx, dy, dz, dvx, dvy, dvz = (
            x_e - x_o,
            y_e - y_o,
            z_e - z_o,
            xe_dot - xo_dot,
            ye_dot - yo_dot,
            ze_dot - zo_dot,
        )

        # 2nd Order CBF
        b = (dx / a1) ** 2 + (dy / a2) ** 2 + (dz / a3) ** 2 - 1.0
        bdot = 2 * (dx * dvx / a1**2 + dy * dvy / a2**2 + dz * dvz / a3**2)
        h = bdot + alpha * b

        return h

    return func


def cbf_grad(obstacle: Array, ellipsoid: List[float], alpha: float) -> Array:
    """Jacobian for the obstacle avoidance constraint function for Fixed-Wing UAV.

    Args:
        x (array-like): concatenated time and state vector
        obstacle (Array): x, y, z location of obstacle
        ellipsoid (List): list of 3D ellipsoid parameters
        alpha (float): class K function (for 2nd order CBF)

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    jacobian = jacfwd(cbf(obstacle, ellipsoid, alpha))

    @jit
    def func(state_and_time: Array) -> Array:
        return jacobian(state_and_time)

    return func


def cbf_hess(obstacle: Array, ellipsoid: List[float], alpha: float) -> Array:
    """Hessian for the obstacle avoidance constraint function for Fixed-Wing UAV.

    Args:
        obstacle (Array): x, y, z location of obstacle
        ellipsoid (List): list of 3D ellipsoid parameters
        alpha (float): class K function (for 2nd order CBF)

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    hessian = jacrev(jacfwd(cbf(obstacle, ellipsoid, alpha)))

    @jit
    def func(state_and_time: Array) -> Array:
        return hessian(state_and_time)

    return func


###############################################################################
# 2nd Order CBF
###############################################################################
obstacle_ho = certificate_package(cbf, cbf_grad, cbf_hess, N)
