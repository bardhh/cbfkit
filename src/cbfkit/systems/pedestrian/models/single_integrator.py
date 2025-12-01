"""
Single Integrator Dynamics for Pedestrian Modeling.

Models a pedestrian as a point mass controlled by acceleration.
State: [px, py, vx, vy]
Control: [ax, ay]
"""

from typing import Tuple

import jax.numpy as jnp
from jax import Array

from cbfkit.utils.user_types import DynamicsCallable


def plant() -> DynamicsCallable:
    """
    Returns a function that computes the dynamics of a single integrator pedestrian model.

    Returns:
        dynamics (DynamicsCallable): function that takes state and returns (f, g)
    """

    def dynamics(x: Array) -> Tuple[Array, Array]:
        """
        Computes f(x) and g(x) for the single integrator system.

        Args:
            x (Array): State vector [px, py, vx, vy]

        Returns:
            Tuple[Array, Array]: f(x) (drift), g(x) (control matrix)
        """
        # x = [px, py, vx, vy]
        # dot_x = [vx, vy, ax, ay]

        # Drift f(x)
        # [vx, vy, 0, 0]
        f = jnp.array([x[2], x[3], 0.0, 0.0])

        # Control matrix g(x)
        # [0, 0]
        # [0, 0]
        # [1, 0]
        # [0, 1]
        g = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        return f, g

    return dynamics
