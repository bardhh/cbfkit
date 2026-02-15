"""
numerical_integration
================

This module contains algorithms for numerically integrating ODEs forward in time.

Functions
---------
forward_euler(x, x_dot, dt): one-step forward-euler discretization
solve_ivp(x, x_dot, dt): uses scipy.integrate.solve_ivp to take one step

Notes
-----
There are plans to implement more generic numerical integration schemes.

This module is, and will continue to be, a work in progress with more functions
added as necessary.

Examples
--------
>>> from cbfkit.utils.numerical_integration import *
>>> import jax.numpy as jnp
>>> x = jnp.array([1.0, 1.0])
>>> x_dot = jnp.array([-0.3, 1.25])
>>> dt = 1e-2
>>> x_new = forward_euler(x, x_dot, dt)
>>> x_new = solve_ivp(x, x_dot, dt)

"""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array


def solve_ivp(x: Array, vector_field: Callable[[Array], Array], dt: float) -> Array:
    """Performs numerical integration on current state (x) using the vector field
    over time interval of length dt according to scipy.integrate.solve_ivp.

    Arguments:
        x: current state
        vector_field: function f(x) -> x_dot
        dt: timestep length (in sec)

    Returns
    -------
        new_state
    """
    from scipy.integrate import solve_ivp as solve

    # Wrap the vector field for scipy (which expects f(t, y) and uses numpy arrays)
    def fun(t, y):
        return np.array(vector_field(jnp.array(y)))

    # Solve
    sol = solve(fun, (0, dt), np.array(x))

    # Return final state
    return jnp.array(sol.y[:, -1])
