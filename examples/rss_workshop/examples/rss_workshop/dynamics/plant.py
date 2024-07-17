import jax.numpy as jnp
from jax import jit, Array, lax
from typing import Optional, Union, Callable
from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns
from .constants import *


def plant(m: float, a: float, b: float, **kwargs) -> DynamicsCallable:
    """
    Returns a function that represents the plant model,
    which computes the drift vector 'f' and control matrix 'g' based on the given state.

    States are the following:
        #! MANUALLY POPULATE

    Control inputs are the following:
        #! MANUALLY POPULATE

    Args:
        perturbation (Optional, Array): additive perturbation to the xdot dynamics
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
        f = jnp.array([x[2], x[3], 0, 0, x[5], 0, 0, 0])
        g = jnp.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [jnp.cos(x[4]) / m, jnp.cos(x[4]) / m, 0, 0],
                [jnp.sin(x[4]) / m, jnp.sin(x[4]) / m, 0, 0],
                [0, 0, 0, 0],
                [
                    (x[1] - x[6]) / (1 / 12 * m * (a**2 + b**2)),
                    (x[1] - x[7]) / (1 / 12 * m * (a**2 + b**2)),
                    0,
                    0,
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        return f, g

    return dynamics
