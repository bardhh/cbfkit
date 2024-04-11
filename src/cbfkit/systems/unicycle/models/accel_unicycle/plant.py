import jax.numpy as jnp
from jax import jit, Array
from typing import Optional, Union


def accel_unicycle_dynamics(sigma: Optional[Union[Array, None]] = None):
    """
    Returns a function that computes the unicycle model dynamics.
    """
    if sigma is not None:
        s = sigma
    else:
        s = jnp.zeros((4, 4))

    @jit
    def dynamics(state):
        """
        Computes the unicycle model dynamics.

        Args:
        state (array-like): The state of the unicycle model, [x, y, v, theta].

        Returns:
        tuple: A tuple containing the function (f) and (g).x
        """
        nonlocal s

        _, _, v, theta = state
        f = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), 0, 0])
        g = jnp.array([[0, 0], [0, 0], [0, 1], [1, 0]])

        return f, g, s

    return dynamics
