import jax.numpy as jnp
from jax import jit, Array, lax
from typing import Optional, Union, Callable
from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns
from .constants import *


def accel_unicycle_dynamics(**kwargs) -> DynamicsCallable:
    """
    Returns a function that represents the plant model,
    which computes the drift vector 'f' and control matrix 'g' based on the given state.

    States are the following:
        x: x-coordinate of unicycle c.o.m. (m)
        y: y-coordinate of unicycle c.o.m. (m)
        v: speed (m/s)
        theta: heading angle (rad)

    Control inputs are the following:
        a: rate of change of speed (m/s^2)
        omega: rate of change of heading angle (rad/s)

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
        f = jnp.array([x[2] * jnp.cos(x[3]), x[2] * jnp.sin(x[3]), 0.0, 0.0])
        g = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        return f, g

    return dynamics
