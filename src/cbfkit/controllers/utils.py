"""Controller utilities.

This module provides utility functions for creating and managing controllers.
"""

import inspect
from typing import Callable, Optional, Tuple, Any

from jax import Array
import jax.numpy as jnp

from cbfkit.utils.user_types import (
    ControllerData,
    Key,
    NominalControllerCallable,
    State,
    Time,
)


def setup_nominal_controller(
    controller_func: Callable[..., Any],
) -> NominalControllerCallable:
    """Adapts a simple controller function into a NominalControllerCallable.

    This helper allows users to define nominal controllers with simpler signatures,
    such as `f(t, x)` or `f(t, x, ref)`, and automatically handles the additional arguments
    required by the simulator (key, ControllerData wrapping).

    Args:
        controller_func: A callable with one of the following signatures:
            - (t, x) -> u
            - (t, x, ref) -> u
            - (t, x, key, ref) -> u (wraps return value only)

    Returns
    -------
        A NominalControllerCallable that can be passed to the simulator.

    Raises
    ------
        ValueError: If the function signature is not supported.
    """
    try:
        sig = inspect.signature(controller_func)
        num_args = len(sig.parameters)
    except ValueError:
        # Fallback for objects that don't support signature
        # We assume 2 arguments as the default simple case.
        num_args = 2

    if num_args == 2:

        def nominal_controller_2(
            t: Time,
            x: State,
            key: Key,
            ref: Optional[State] = None,
        ) -> Tuple[Array, ControllerData]:
            u = controller_func(t, x)
            return u, ControllerData(u_nom=u)

        return nominal_controller_2

    elif num_args == 3:

        def nominal_controller_3(
            t: Time,
            x: State,
            key: Key,
            ref: Optional[State] = None,
        ) -> Tuple[Array, ControllerData]:
            u = controller_func(t, x, ref)
            return u, ControllerData(u_nom=u)

        return nominal_controller_3

    elif num_args == 4:

        def nominal_controller_4(
            t: Time,
            x: State,
            key: Key,
            ref: Optional[State] = None,
        ) -> Tuple[Array, ControllerData]:
            ret = controller_func(t, x, key, ref)
            if isinstance(ret, tuple) and len(ret) == 2:
                # Assume it's (u, data)
                return ret  # type: ignore
            return ret, ControllerData(u_nom=ret)

        return nominal_controller_4

    else:
        raise ValueError(
            f"Unsupported nominal controller signature with {num_args} arguments. "
            "Expected (t, x), (t, x, ref), or (t, x, key, ref)."
        )
