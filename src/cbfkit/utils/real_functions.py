"""
real_functions
================

This file contains a collection of real functions.

Functions
---------
tanh_sigmoid(s, sbar, k): uses a form of hyperbolic tangent for a sigmoid function

Notes
-----
This module is, and will continue to be, a work in progress with more functions
added as necessary.

Examples
--------
>>> from real_functions import *
>>> val = tanh_sigmoid_func(2.7, 2.0)


"""

import jax.numpy as jnp
from jax import jit


@jit
def tanh_sigmoid(s: float, sbar: float, k: float = 100.0) -> float:
    """Computes the value of the hyperbolic tangent sigmoid function.

    Args:
        s (float): input to sigmoid function
        sbar (float): maximum value
        k (float, optional): gain. Defaults to 100.0.

    Returns:
        float: value of function evaluated
    """
    return s * (1 / 2 + 1 / 2 * jnp.tanh(k * s)) + (sbar - s) * (
        1 / 2 + 1 / 2 * jnp.tanh(k * (s - sbar))
    )
