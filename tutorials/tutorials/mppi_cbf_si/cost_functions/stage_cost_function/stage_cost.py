"""
#! MANUALLY POPULATE (docstring)
"""
import jax.numpy as jnp
from jax import jit, Array, lax
from typing import List, Callable

N = 2


###############################################################################
# Stage Cost Function
###############################################################################


def stage_cost(
    goal: float, obstacle: float, obstacle_radius: float, **kwargs
) -> Callable[[Array, Array], Array]:
    """Super-level set convention.

    Args:
        #! kwargs -- optional to manually populate

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """

    @jit
    def func(state: Array, action: Array) -> Array:
        """Function to be evaluated.

        Args:
            state (Array): state vector

        Returns:
            Array: cost value
        """
        x = state
        dist = jnp.linalg.norm(x[0:2] - obstacle[0:2]) - obstacle_radius
        return 0.2 * ((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2) + 10.0 / jnp.maximum(
            dist, 0.01
        )

    return func
