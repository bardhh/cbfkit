import jax.numpy as jnp
from jax import jit, Array
from typing import Optional, Union, Callable


def approx_unicycle_dynamics(l: float = 1.0):
    """
    Returns a function that represents the approximate unicycle dynamics, which computes
    the drift vector 'f' and control matrix 'g' based on the given state.

    Taken from R. Olfati-Saber, "Near-identity diffeomorphisms and exponential e-tracking
    and 6-stabilization of first-order nonholonomic SE (2) vehicles", 2002.
    """

    @jit
    def dynamics(state: Array):
        """
        Computes the drift vector 'f' and control matrix 'g' based on the given state.

        :param state: A numpy array representing the current state (x, y, theta, l)
                      where x and y are positions, theta is the orientation angle,
                      and l is the wheelbase of the unicycle.
        :return: A tuple (f, g) where f is the drift vector and g is the control matrix.
        """

        _, _, theta = state
        f = jnp.array([0, 0, 0])
        g = jnp.array(
            [
                [jnp.cos(theta), -l * jnp.sin(theta)],
                [jnp.sin(theta), l * jnp.cos(theta)],
                [0, 1],
            ]
        )

        return f, g

    return dynamics
