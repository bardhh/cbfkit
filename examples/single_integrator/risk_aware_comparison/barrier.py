"""Circular keep-in barrier in both conventions.

- ``make_barrier``: reframed zeroing barrier ``h(x) = 1 - ||x||^2/R_c^2`` (safe set ``{h > 0}``),
  used by the per-step stochastic CBF (``h >= 0`` convention).
- ``make_cost_barrier``: cost barrier ``B(x) = ||x||^2/R_c^2`` (safe set ``{B < 1}``), used by the
  accumulating path-integral RA-CBF (``B`` convention).
Same physical safe set ``{||x|| < R_c}``; ``B = 1 - h``.
"""
import jax.numpy as jnp
from jax import Array


def make_barrier(r_c: float):
    def h(x: Array) -> Array:
        return 1.0 - jnp.dot(x, x) / (r_c**2)

    return h


def make_cost_barrier(r_c: float):
    def b(x: Array) -> Array:
        return jnp.dot(x, x) / (r_c**2)

    return b
