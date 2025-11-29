"""
Ellipsoidal Obstacle Scenario for Unicycle

This module provides barrier functions for avoiding ellipsoidal obstacles.
It now uses the core `cbfkit` library for the generic barrier logic.
"""

from typing import Callable

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.certificates import certificate_package
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory

N = 4  # State dimension [x, y, v, theta]

# Generate the standard barrier functions using the core factory
# Unicycle state: [x, y, v, theta] -> pos indices (0, 1)
# Obstacle state: [x_o, y_o, t] -> pos indices (0, 1)
# Ellipsoid axes: [a1, a2] -> indices (0, 1)
cbf, cbf_grad, cbf_hess = ellipsoidal_barrier_factory(
    system_position_indices=(0, 1), obstacle_position_indices=(0, 1), ellipsoid_axis_indices=(0, 1)
)


def stochastic_cbf(obstacle: Array, ellipsoid: Array) -> Callable[[Array], Array]:
    """
    Stochastic version (High Probability Safety).
    Uses exponential transformation.
    """

    @jit
    def func(state_and_time: Array) -> Array:
        x_e, y_e, _v_e, _theta_e, _t = state_and_time
        x_o, y_o, _t = obstacle
        a1, a2 = ellipsoid

        b = ((x_e - x_o) / (a1)) ** 2 + ((y_e - y_o) / (a2)) ** 2 - 1.0
        # Note: This formulation seems specific to specific noise models/theorems
        return jnp.exp(-0.1 * b)

    return func


# Exportable package
obstacle_ca = certificate_package(cbf, cbf_grad, cbf_hess, N)
