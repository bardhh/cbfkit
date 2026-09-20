"""future_focused.py.

Contains functions defining a future-focused CBF wrt an ellipsoidal obstacle.

Exportable:
    obstacle_ff
"""

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit

from cbfkit.certificates import certificate_package
from cbfkit.utils.real_functions import tanh_sigmoid

N = 6


#! TO DO: Generalize these types of functions to be model-agnostic
###############################################################################
# Future-Focused CBF (Constant Velocity)
###############################################################################


@jit
def cbf(state: Array, *, obstacle: Array, r: float, tfuture: float) -> Array:
    """Obstacle avoidance constraint function for Fixed-Wing UAV. Super-level set convention.

    Args:
        x (Array): concatenated time and state vector

    KwArgs:
        obstacle (Array): state vector for obstacle
        r (float): radius of obstacle
        tfuture (float): maximum lookahead time

    Returns
    -------
        ret (float): value of constraint function evaluated at time and state
    """
    # Get states
    x_e, y_e, z_e, v_e, psi_e, gamma_e, _t = state
    x_o, y_o, z_o, v_o, psi_o, gamma_o, _ = obstacle

    # Ego Velocities
    xe_dot = v_e * jnp.cos(psi_e) * jnp.cos(gamma_e)
    ye_dot = v_e * jnp.sin(psi_e) * jnp.cos(gamma_e)
    ze_dot = v_e * jnp.sin(gamma_e)

    # Constant obstacle locations
    xo_dot = v_o * jnp.cos(psi_o) * jnp.cos(gamma_o)
    yo_dot = v_o * jnp.sin(psi_o) * jnp.cos(gamma_o)
    zo_dot = v_o * jnp.sin(gamma_o)

    # FF-CBF
    dx, dy, dz, dvx, dvy, dvz = (
        x_e - x_o,
        y_e - y_o,
        z_e - z_o,
        xe_dot - xo_dot,
        ye_dot - yo_dot,
        ze_dot - zo_dot,
    )
    tau_hat = -(dx * dvx + dy * dvy + dz * dvz) / (dvx**2 + dvy**2 + dvz**2 + 1e-3)
    tau = tanh_sigmoid(tau_hat, tfuture)

    h = (
        dx**2
        + dy**2
        + dz**2
        + 2 * tau * (dx * dvx + dy * dvy + dz * dvz)
        + tau**2 * (dvx**2 + dvy**2 + dvz**2)
        - r**2
    )

    return h


@jit
def cbf_grad(state: Array, *, obstacle: Array, r: float, tfuture: float) -> Array:
    """Jacobian for the obstacle avoidance constraint function for Fixed-Wing UAV.

    Args:
        x (array-like): concatenated time and state vector

    KwArgs:
        obstacle (Array): state vector for obstacle
        r (float): radius of obstacle
        tfuture (float): maximum lookahead time

    Returns
    -------
        ret (float): value of Jacobian evaluated at time and state
    """
    return jacfwd(cbf)(state, obstacle, r, tfuture)


@jit
def cbf_hess(state: Array, *, obstacle: Array, r: float, tfuture: float) -> Array:
    """Hessian for the obstacle avoidance constraint function for Fixed-Wing UAV.

    Args:
        x (array-like): concatenated time and state vector

    KwArgs:
        obstacle (Array): state vector for obstacle
        r (float): radius of obstacle
        tfuture (float): maximum lookahead time

    Returns
    -------
        ret (float): value of Hessian evaluated at time and state
    """
    return jacrev(jacfwd(cbf))(state, obstacle, r, tfuture)


###############################################################################
# Future-Focused CBF (Constant Velocity)
###############################################################################
obstacle_ff = certificate_package(cbf, cbf_grad, cbf_hess, N)
