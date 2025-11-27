import jax.numpy as jnp
from typing import *
from jax import jit, Array, lax
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns, Key


def controller_1(
    kp: float,
    xg: float,
    yg: float,
) -> ControllerCallable:
    """
    Create a controller for the given dynamics.

    Args:
        #! USER-POPULATE

    Returns:
        controller (Callable): handle to function computing control

    """

    @jit
    def controller(t: float, x: Array, key: Key, xd: float) -> ControllerCallableReturns:
        """Computes control input (2x1).

        Args:
            t (float): time in sec
            x (Array): state vector (or estimate if using observer/estimator)

        Returns:
            unom (Array): 2x1 vector
            data: (dict): empty dictionary
        """
        # logging data
        u_nom = -kp * (x[0] - xg), -kp * (x[1] - yg)
        data = {"u_nom": u_nom}

        return jnp.array(u_nom), data

    return controller
