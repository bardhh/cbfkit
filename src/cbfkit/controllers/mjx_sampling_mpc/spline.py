"""Control-spline interpolation: knots ``(B, K, nu)`` -> controls ``(B, H, nu)``.

Only zero-order hold and linear are provided; both are dependency-free. Cubic
would need ``interpax`` (what hydrax uses) and is deliberately out of scope.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

InterpFunc = Callable[[Array, Array, Array], Array]


def interp_zero(tq: Array, tk: Array, knots: Array) -> Array:
    """Zero-order hold: each query time takes the most recent knot (clamped at both ends)."""
    idx = jnp.searchsorted(tk, tq, side="right") - 1
    idx = jnp.clip(idx, 0, tk.shape[0] - 1)
    return knots[:, idx, :]


def interp_linear(tq: Array, tk: Array, knots: Array) -> Array:
    """Piecewise-linear interpolation, batched over ``B`` and over control channels."""

    def _channel(k: Array) -> Array:  # k: (K,)
        return jnp.interp(tq, tk, k)  # (H,)

    def _one(kb: Array) -> Array:  # kb: (K, nu)
        return jax.vmap(_channel, in_axes=1, out_axes=1)(kb)  # (H, nu)

    return jax.vmap(_one)(knots)  # (B, H, nu)


def get_interp_func(name: str) -> InterpFunc:
    if name == "zero":
        return interp_zero
    if name == "linear":
        return interp_linear
    raise ValueError(
        f"Unknown spline_type {name!r}; expected 'zero' or 'linear' ('cubic' is not supported)."
    )
