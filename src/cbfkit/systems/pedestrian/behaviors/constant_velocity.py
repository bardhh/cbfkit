"""
Constant Velocity Policy for Pedestrians.

This policy models a pedestrian who continues to move at their current velocity,
applying zero acceleration.
"""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import Array, jit

# Assuming a policy interface: (t, state, environment_state) -> control_input
# For constant velocity, environment_state is not used.
# Control input u = [ax, ay]


def policy() -> Callable[[float, Array, Tuple[Array, ...]], Array]:
    """
    Returns a callable policy function for a constant velocity pedestrian.

    The policy function takes the current time, pedestrian state, and environment state
    (which can include other agents' states, goals, etc.) and returns a control input
    (acceleration) for the pedestrian.

    For constant velocity, the control input (acceleration) is always zero.

    Args:
        None

    Returns:
        Callable[[float, Array, Tuple[Array, ...]], Array]: A policy function that
            returns a zero acceleration vector `[0.0, 0.0]`.
    """

    @jit
    def constant_velocity_policy(
        t: float,  # Current time
        state: Array,  # Pedestrian state [px, py, vx, vy]
        environment_state: Tuple[Array, ...],  # Other agents' states, goals, etc.
    ) -> Array:
        """
        Policy implementation for constant velocity. Always returns zero acceleration.
        """
        # A pedestrian modeled as a single integrator (px, py, vx, vy)
        # needs an acceleration input (ax, ay). For constant velocity, ax=0, ay=0.
        return jnp.array([0.0, 0.0])

    return constant_velocity_policy
