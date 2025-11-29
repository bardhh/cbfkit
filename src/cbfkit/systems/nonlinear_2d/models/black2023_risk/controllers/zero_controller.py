from typing import Any, Dict, Optional

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import (
    Control,
    ControllerCallable,
    ControllerCallableReturns,
    ControllerData,
    Key,
    State,
)


def zero_controller() -> ControllerCallable:
    """
    Create a zero controller for the given fixed-wing uav dynamics.

    Args:
        None

    Returns:
        controller (Callable): handle to function computing zero control

    """

    @jit
    def controller(
        _t: float,
        _state: State,
        _u_nom: Optional[Control] = None,
        _key: Optional[Key] = None,
        _data: Optional[ControllerData] = None,
    ) -> ControllerCallableReturns:
        """Computes zero control input (1x1).

        Args:
            _t (float): time in sec
            _state (Array): state vector (or estimate if using observer/estimator)

        Returns:
            zeros (Array): 1x1 zero vector
            data: (dict): empty dictionary
        """
        # logging data
        data: Dict[str, Any] = {}

        return jnp.zeros((2,)), ControllerData(sub_data=data)

    return controller
