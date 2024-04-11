import jax.numpy as jnp
from jax import Array


def naive(_t: float, y: Array, _z: Array, _u: Array, _p: Array) -> Array:
    """Naive estimator -- returns the exact measurement value as the estimated state.

    Args:
        t (float): time (sec) -- unused but required for generic sensor
        y (Array): state measurement (assumed full state)
        z (Array): state estimate vector -- unused but required for generic sensor
        u (Array): control input vector -- unused but required for generic sensor
        p (Array): state estimate covariance vector -- unused but required for generic sensor

    Returns:
        y (Array): naively returns the (full) state measurement

    """
    return y, jnp.zeros((len(y), len(y)))
