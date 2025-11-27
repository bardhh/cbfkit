import jax.numpy as jnp
from jax import jit, Array
from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns
from .constants import *


def plant(**kwargs) -> DynamicsCallable:
    """
    Returns a function that represents the plant model,
    which computes the drift vector 'f' and control matrix 'g' based on the given state.

    States are the following:
        #! MANUALLY POPULATE

    Control inputs are the following:
        #! MANUALLY POPULATE

    Args:
        kwargs: keyword arguments

    Returns:
        dynamics (Callable): takes state as input and returns dynamics components
            f, g of the form dx/dt = f(x) + g(x)u

    """

    @jit
    def dynamics(x: Array) -> DynamicsCallableReturns:
        """
        Computes the drift vector 'f' and control matrix 'g' based on the given state x.

        Args:
            x (Array): state vector

        Returns:
            dynamics (DynamicsCallable): takes state as input and returns dynamics components f, g
        """
        f = jnp.array([x[3], x[4], x[5], 0, 0, 0, x[9], x[10], x[11], 0, 0, 0])
        g = jnp.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return f, g

    return dynamics
