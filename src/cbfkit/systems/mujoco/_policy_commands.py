"""Shared normalization for scheduled and state-aware torso commands."""

import inspect
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import jax.numpy as jnp
from jax import Array

TorsoCommand = Optional[Union[Sequence[float], Array, Callable[..., Array]]]
TorsoCommandCallable = Callable[[float, Array, Mapping[str, Any]], Array]


def normalize_torso_command(command: TorsoCommand, size: int) -> TorsoCommandCallable:
    """Adapt constant, time-only, or ``(t, x, sub_data)`` commands once at setup."""
    if command is None:
        constant = jnp.zeros(size)
    elif callable(command):
        signature = inspect.signature(command)
        try:
            signature.bind(0.0, None, {})
        except TypeError:
            try:
                signature.bind(0.0)
            except TypeError as exc:
                raise ValueError("torso command must accept (t) or (t, x, sub_data)") from exc

            def scheduled(t: float, x: Array, sub: Mapping[str, Any]) -> Array:
                return command(t)

            return scheduled
        return command
    else:
        constant = jnp.asarray(command, dtype=float)

    def fixed(t: float, x: Array, sub: Mapping[str, Any]) -> Array:
        return constant

    return fixed
