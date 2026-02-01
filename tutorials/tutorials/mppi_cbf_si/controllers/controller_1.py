import jax.numpy as jnp
from typing import *
from jax import jit, Array, lax
from cbfkit.utils.user_types import (
    NominalControllerCallable,
    ControllerCallableReturns,
    Key,
    ControllerData,
)


def controller_1(
    k_p: float,
) -> NominalControllerCallable:
    """
    Create a controller for the given dynamics.

    Args:
        #! USER-POPULATE

    Returns:
        controller (Callable): handle to function computing control

    """

    @jit
    def controller(t: float, x: Array, key: Key, xd: Optional[Array]) -> ControllerCallableReturns:
        """Computes control input.

        Args:
            t (float): time in sec
            x (Array): state vector
            key (Key): random key
            xd (Optional[Array]): desired state

        Returns:
            unom (Array): vector
            data: (ControllerData):
        """
        # logging data
        u_nom_val = -k_p * (x[0] - xd[0]), -k_p * (x[1] - xd[1])
        data = ControllerData(u_nom=u_nom_val)

        return jnp.array(u_nom_val), data

    return controller
