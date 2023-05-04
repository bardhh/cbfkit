import jax.numpy as jnp


def single_integrator(inputs):
    def dynamics(state):
        # Define the constants
        f = jnp.array([0, 0])
        g = jnp.array([[1, 0], [0, 1]])
        return f, g

    return dynamics
