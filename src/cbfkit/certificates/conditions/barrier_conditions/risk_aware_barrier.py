"""
risk_aware_barrier.py
================

This module contains the function specifying the right-hand side of risk-aware CBF inequalities,
e.g., incomplete -- risk margin derivation pending.

Functions
---------
-right_hand_side(alpha): -alpha * h + r

Notes
-----
These functions are typically used as arguments to a CertificatePackage object.

Examples
--------
>>> from jax import jacfwd, jacrev
>>> from cbfkit.certificates import certificate_package, concatenate_certificates
>>> from cbfkit.certificates.conditions.barrier_conditions.risk_aware_barrier import right_hand_side
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


def right_hand_side(rho: float, alpha: float) -> Callable[[Array], Array]:
    """Generates function for computing RHS of barrier conditions for stochastic CBF:

    hdot <= -alpha*h + r

    where ``r`` is the risk-aware buffer term derived from rho. The derivation of ``r``
    is incomplete (the original implementation hardcoded ``r = 0.0``, which collapses
    this condition to a vanilla zeroing CBF with no risk margin).

    For risk-aware safety conditions today, use
    :func:`cbfkit.certificates.conditions.barrier_conditions.path_integral_barrier.right_hand_side`,
    which has a complete derivation.

    Args:
        rho (float): tolerable risk of violating constraint
        alpha (float): class-K gain

    Raises
    ------
        NotImplementedError: always, since the risk margin term ``r`` is undefined.
    """
    assert alpha >= 0
    assert 0 < rho < 1

    raise NotImplementedError(
        "risk_aware_barrier.right_hand_side is not implemented: the risk-margin term "
        "'r' has not been derived. Using r=0 silently degrades the condition to a "
        "vanilla zeroing CBF with no risk guarantee. "
        "Use path_integral_barrier.right_hand_side for a complete risk-aware barrier."
    )
