"""
Runge-Kutta Integration
=======================

This module contains Runge-Kutta numerical integration schemes.

Functions
---------
runge_kutta_4(x, vector_field, dt): Classical RK4 integration step.

"""

from typing import Callable

from jax import Array, jit


def runge_kutta_4(x: Array, vector_field: Callable[[Array], Array], dt: float) -> Array:
    """Performs numerical integration on current state (x) using the vector field
    over time interval of length dt according to the classical Runge-Kutta 4 method.

    Arguments:
        x: current state
        vector_field: function f(x) -> x_dot
        dt: timestep length (in sec)

    Returns
    -------
        new_state
    """
    k1 = vector_field(x)
    k2 = vector_field(x + 0.5 * dt * k1)
    k3 = vector_field(x + 0.5 * dt * k2)
    k4 = vector_field(x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
