from typing import Optional, Tuple, Union

import jax.numpy as jnp

from cbfkit.utils.user_types import Array, Covariance, State, Time


def naive(
    _t: Time,
    y: Array,
    _z: State,
    _u: Optional[Union[Array, None]] = None,
    _c: Optional[Union[Covariance, None]] = None,
) -> Tuple[State, Covariance]:
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
