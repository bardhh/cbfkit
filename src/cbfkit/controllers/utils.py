"""Controller utilities.

This module provides utility functions for creating and managing controllers.
"""

import inspect
from typing import Callable, Optional, Tuple, Any

from jax import Array

from cbfkit.utils.user_types import (
    ControllerCallable,
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


def _normalize_controller_return(
    ret: Any,
    u_nom: Optional[Array],
) -> Tuple[Array, ControllerData]:
    """Normalize controller outputs to ``(u, ControllerData)``."""
    if isinstance(ret, tuple) and len(ret) == 2:
        u, data = ret
        if isinstance(data, ControllerData):
            return u, data
        if isinstance(data, dict):
            return u, ControllerData(**data)
        return u, ControllerData(u=u, u_nom=u_nom)

    return ret, ControllerData(u=ret, u_nom=u_nom)


def setup_controller(controller_func: Callable[..., Any]) -> ControllerCallable:
    """Adapts legacy controller signatures to ``ControllerCallable``.

    Supported input signatures:
    - ``(t, x) -> u`` or ``(t, x) -> (u, data)``
    - ``(t, x, u_nom) -> u`` or ``(t, x, u_nom) -> (u, data)``
    - ``(t, x, u_nom, key) -> u`` or ``(t, x, u_nom, key) -> (u, data)``
    - ``(t, x, key, data) -> u`` or ``(t, x, key, data) -> (u, data)``
    - ``(t, x, u_nom, key, data) -> u`` or ``(t, x, u_nom, key, data) -> (u, data)``

    Returns:
        ControllerCallable-compatible wrapper.
    """
    if getattr(controller_func, "__cbfkit_controller_adapter__", False):
        return controller_func  # type: ignore[return-value]

    try:
        sig = inspect.signature(controller_func)
        params = list(sig.parameters.values())
        num_args = len(params)
        param_names = [p.name.lower() for p in params]
    except ValueError:
        num_args = 5
        param_names = []

    if num_args == 2:

        def wrapped_controller_2(
            t: Time,
            x: State,
            u_nom: Array,
            key: Key,
            data: ControllerData,
        ) -> Tuple[Array, ControllerData]:
            ret = controller_func(t, x)
            return _normalize_controller_return(ret, u_nom)

        wrapped_controller_2.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
        return wrapped_controller_2

    if num_args == 3:

        def wrapped_controller_3(
            t: Time,
            x: State,
            u_nom: Array,
            key: Key,
            data: ControllerData,
        ) -> Tuple[Array, ControllerData]:
            ret = controller_func(t, x, u_nom)
            return _normalize_controller_return(ret, u_nom)

        wrapped_controller_3.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
        return wrapped_controller_3

    if num_args == 4:
        # Compatibility for both legacy forms:
        # - (t, x, u_nom, key)
        # - (t, x, key, data)
        third = param_names[2] if len(param_names) >= 3 else ""
        fourth = param_names[3] if len(param_names) >= 4 else ""
        treat_as_key_data = ("key" in third) and ("data" in fourth or fourth == "d")

        if treat_as_key_data:

            def wrapped_controller_4_key_data(
                t: Time,
                x: State,
                u_nom: Array,
                key: Key,
                data: ControllerData,
            ) -> Tuple[Array, ControllerData]:
                ret = controller_func(t, x, key, data)
                return _normalize_controller_return(ret, u_nom)

            wrapped_controller_4_key_data.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
            return wrapped_controller_4_key_data

        def wrapped_controller_4_u_key(
            t: Time,
            x: State,
            u_nom: Array,
            key: Key,
            data: ControllerData,
        ) -> Tuple[Array, ControllerData]:
            ret = controller_func(t, x, u_nom, key)
            return _normalize_controller_return(ret, u_nom)

        wrapped_controller_4_u_key.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
        return wrapped_controller_4_u_key

    if num_args == 5:

        def wrapped_controller_5(
            t: Time,
            x: State,
            u_nom: Array,
            key: Key,
            data: ControllerData,
        ) -> Tuple[Array, ControllerData]:
            ret = controller_func(t, x, u_nom, key, data)
            return _normalize_controller_return(ret, u_nom)

        wrapped_controller_5.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
        return wrapped_controller_5

    raise ValueError(
        f"Unsupported controller signature with {num_args} arguments. "
        "Expected (t, x), (t, x, u_nom), (t, x, u_nom, key), "
        "(t, x, key, data), or (t, x, u_nom, key, data)."
    )
