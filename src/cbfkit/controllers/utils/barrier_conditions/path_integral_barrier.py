"""
path_integral_barrier.py
================

This module contains the function specifying the right-hand side of risk-aware path-integral 
CBF inequalities, e.g., 1 - gamma - sqrt(2 * T) * eta * erfinv(1 - rho) + integral for 
Lh <= 1 - gamma - sqrt(2 * T) * eta * erfinv(1 - rho) + integral.

Functions
---------
-right_hand_side(rho, gamma, eta, time_period): 1 - gamma - sqrt(2 * T) * eta * erfinv(1 - rho) + integral

Notes
-----
These functions are typically used as arguments to a CertificatePackage object.

Examples
--------
>>> from jax import jacfwd, jacrev
>>> from cbfkit.controllers.utils.certificate_packager import certificate_package, concatenate_certificates
>>> from cbfkit.controllers.utils.barrier_conditions.path_integral_barrier import right_hand_side
>>> 
>>> def cbf(limit):
>>>     def func(x):
>>>         return x[0] - limit
>>>     return func
>>>
>>> def cbf_grad(limit):
>>>     jacobian = jacfwd(cbf(limit))
>>>     def func(x):
>>>         return jacobian(x)
>>>     return func
>>>
>>> def cbf_hess(limit):
>>>     hessian = jacrev(jacfwd(cbf(limit)))
>>>     def func(x):
>>>         return hessian(x)
>>>     return func
>>> 
>>> package = certificate_package(cbf, cbf_grad, cbf_hess, n=1)
>>>
>>> limit = 1.0
>>> rho, gamma, eta, time_period = 1.0, 0.5, 2.0, 1.0
>>> stochastic_barriers = concatenate_certificates(
>>>     package(certificate_conditions=right_hand_side(rho, gamma, eta, time_period), limit=limit), 
>>> )

"""

from typing import Callable
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import erfinv


def right_hand_side(
    rho: float, gamma: float, eta: float, time_period: float
) -> Callable[[Array], Array]:
    """Generates function for computing RHS of barrier conditions for risk-aware path integral CBF:

    Lh <= 1 - gamma - sqrt(2 * T) * eta * erfinv(1 - rho) + integral

    where integral is the Lebesgue integral of the generator of h from 0 to the current time.

    Args:
        rho (float): tolerable risk of violating constraint
        gamma (float): maximum initial value h, i.e., sup_{x0 in X0}h(x0)
        eta (float): maximum value of dh/dx * sigma(x) in constraint set, i.e., sup_{x in S}||dh/dx * sigma(x)||
        time_period (float): length of time interval of system operation (in sec)

    Returns:
        Callable[[Array], Array]: Risk-Aware Path Integral CBF barrier condition
    """
    assert 0 < rho < 1
    assert gamma > 0
    assert eta > 0
    assert 0 < time_period < jnp.inf

    return lambda integral: 1 - gamma - jnp.sqrt(2 * time_period) * eta * erfinv(1 - rho) + integral
