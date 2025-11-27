"""
Ellipsoidal Obstacle Scenario for Unicycle

This module provides barrier functions for avoiding ellipsoidal obstacles.
It is decoupled from the core `cbfkit` library to serve as a modular example.
"""

from jax import jit, jacfwd, jacrev, Array
from cbfkit.certificates import certificate_package
import jax.numpy as jnp

N = 4  # State dimension [x, y, v, theta]

def cbf(obstacle: Array, ellipsoid: Array) -> Array:
    """
    Obstacle avoidance constraint function.
    Super-level set convention (h(x) >= 0 is safe).
    
    Args:
        obstacle (Array): [x_o, y_o, t] state of obstacle
        ellipsoid (Array): [a1, a2] semi-axes of ellipsoid
    """
    @jit
    def func(state_and_time: Array) -> Array:
        x_e, y_e, _v_e, _theta_e, _t = state_and_time
        x_o, y_o, _t = obstacle
        a1, a2 = ellipsoid

        # Ellipsoid equation: ((x-xo)/a1)^2 + ((y-yo)/a2)^2 - 1 >= 0 (outside)
        # Note: Original implementation seemed to use <= 0 for collision?
        # Let's verify. If b > 0, we are outside. If b < 0, we are inside.
        # ZCBF requires h(x) >= 0.
        # So b as defined below is correct.
        
        b = ((x_e - x_o) / (a1)) ** 2 + ((y_e - y_o) / (a2)) ** 2 - 1.0
        return b

    return func

def cbf_grad(obstacle: Array, ellipsoid: Array) -> Array:
    """Jacobian via Auto-diff."""
    jacobian = jacfwd(cbf(obstacle, ellipsoid))
    @jit
    def func(state_and_time: Array) -> Array:
        return jacobian(state_and_time)
    return func

def cbf_hess(obstacle: Array, ellipsoid: Array) -> Array:
    """Hessian via Auto-diff."""
    hessian = jacrev(jacfwd(cbf(obstacle, ellipsoid)))
    @jit
    def func(state_and_time: Array) -> Array:
        return hessian(state_and_time)
    return func

def stochastic_cbf(obstacle: Array, ellipsoid: Array) -> Array:
    """
    Stochastic version (High Probability Safety).
    Uses exponential transformation.
    """
    @jit
    def func(state_and_time: Array) -> Array:
        x_e, y_e, _v_e, _theta_e, _t = state_and_time
        x_o, y_o, _t = obstacle
        a1, a2 = ellipsoid

        b = ((x_e - x_o) / (a1)) ** 2 + ((y_e - y_o) / (a2)) ** 2 - 1.0
        # Note: This formulation seems specific to specific noise models/theorems
        return jnp.exp(-0.1 * b) 

    return func

# Exportable package
obstacle_ca = certificate_package(cbf, cbf_grad, cbf_hess, N)
