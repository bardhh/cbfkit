import jax.numpy as jnp
from jax import jit


def zero_controller():
    """
    Create a zero controller for the given unicycle dynamics.

    :param dynamics: approximate unicycle dynamics ode
    :return: A function that computes control inputs based on the current state and desired state.
    """

    @jit
    def controller(_t, _state):
        # logging data
        data = {}

        return jnp.zeros((2,)), data

    return controller
