import jax.numpy as jnp
from jax import Array, jit


def two_dimensional_single_integrator(**kwargs):
    """
    Returns a function that represents the single integrator dynamics.
    x_dot = u
    """

    @jit
    def dynamics(state: Array):
        f = jnp.zeros(2)
        g = jnp.eye(2)
        return f, g

    return dynamics
