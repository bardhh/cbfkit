import jax.numpy as jnp
from jax.scipy.special import exp


def careful_agent(inputs, radius, multiplier):
    def dynamics(state):
        # Define the constants
        c = multiplier

        # Compute the changes in x, y, and theta
        delta_x = (state[0] - inputs[0]) * exp(c * (radius - (state[0] - inputs[0]) ** 2))
        delta_y = (state[1] - inputs[1]) * exp(c * (radius - (state[1] - inputs[1]) ** 2))
        delta_theta = 1 / (1 + (delta_y / delta_x) ** 2)

        # Return the changes as a JAX array
        return jnp.array([delta_x, delta_y, delta_theta])

    return dynamics


def careless_agent(inputs):
    def dynamics(state):
        # Return the inputs as the changes in x, y, and theta for the careless agent
        return jnp.array(inputs)

    return dynamics
