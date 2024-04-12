"""
risk_aware_barrier.py
================

This module contains the function specifying the right-hand side of risk-aware CBF inequalities,
e.g., UNFINSHED.

Functions
---------
-right_hand_side(alpha): -alpha * h + r

Notes
-----
These functions are typically used as arguments to a CertificatePackage object.

Examples
--------
>>> from jax import jacfwd, jacrev
>>> from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import certificate_package, concatenate_certificates
>>> from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions.risk_aware_barrier import right_hand_side
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
>>> alpha, beta = 1.0, 0.5
>>> stochastic_barriers = concatenate_certificates(
>>>     package(certificate_conditions=right_hand_side(alpha, beta), limit=limit), 
>>> )

"""

from typing import Callable
from jax import Array

#! INCOMPLETE -- STILL NEED TO DERIVE CONDITIONS


def right_hand_side(rho: float, alpha: float) -> Callable[[Array], Array]:
    """Generates function for computing RHS of barrier conditions for stochastic CBF:

    hdot <= -alpha*h + beta

    Args:
        None

    Returns:
        Callable[[Array], Array]: Zeroing CBF barrier conditions
    """
    assert alpha >= 0
    assert 0 < rho < 1

    #! Define this quantity
    r = 0.0
    return lambda h: -alpha * h + r
