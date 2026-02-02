"""
#! MANUALLY POPULATE (docstring)
"""
import jax.numpy as jnp
from jax import jit, Array
from typing import Callable

###############################################################################
# Terminal Cost Function
###############################################################################


def terminal_cost(
    goal: Array, obstacle: Array, obstacle_radius: float, **kwargs
) -> Callable[[Array, Array], Array]:
    """Super-level set convention.

    Args:
        #! kwargs -- optional to manually populate

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """

    @jit
    def func(state_and_time: Array, action: Array) -> Array:
        """Function to be evaluated.

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: cbf value
        """
        x = state_and_time
        return 0.2 * ((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2) + 10.0 / jnp.maximum(
            (jnp.linalg.norm(x[0:2] - obstacle[0:2]) - obstacle_radius), 0.01
        )

    return func
