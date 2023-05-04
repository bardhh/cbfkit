import jax.numpy as jnp


def nimble_ant_dynamics():
    def dynamics(state):
        f = jnp.array([[0], [0]])
        g = jnp.array([[1], [1]])
        return f, g

    return dynamics
