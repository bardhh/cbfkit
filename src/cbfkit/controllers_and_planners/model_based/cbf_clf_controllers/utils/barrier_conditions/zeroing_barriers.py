"""
zeroing_barriers.py
================

This module contains a library of functions specifying the right-hand side of zeroing CBF inequalities,
e.g., alpha(h) for hdot >= -alpha(h).

Functions
---------
-linear_class_k(alpha): alpha * h
-cubic_class_k(alpha): alpha * h^3
-generic_class_k(alpha): alpha(h)

Notes
-----
These functions are typically used as arguments to a CertificatePackage object.

Examples
--------
>>> from jax import jacfwd, jacrev
>>> from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import certificate_package, concatenate_certificates
>>> from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.barrier_conditions.zeroing_barriers import linear_class_k
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
>>> alpha = 1.0
>>> barriers = concatenate_certificates(
>>>     package(certificate_conditions=linear_class_k(alpha), limit=limit), 
>>> )

"""

from typing import Callable
from jax import Array


def linear_class_k(alpha: float) -> Callable[[Array], Array]:
    """Generates function for computing RHS of barrier conditions for zeroing CBF:

    hdot >= -alpha*h

    Args:
        None

    Returns:
        Callable[[Array], Array]: Zeroing CBF barrier conditions
    """
    assert alpha >= 0
    return lambda h: alpha * h


def cubic_class_k(alpha: float) -> Callable[[Array], Array]:
    """Generates function for computing RHS of barrier conditions for zeroing CBF:

    hdot >= -alpha*h**3

    Args:
        None

    Returns:
        Callable[[Array], Array]: Zeroing CBF barrier conditions
    """
    assert alpha >= 0
    return lambda h: alpha * h**3


def generic_class_k(alpha: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Generates function for computing RHS of barrier conditions for zeroing CBF:

    hdot >= -alpha(h)

    Args:
        None

    Returns:
        Callable[[Array], Array]: Zeroing CBF barrier conditions
    """

    def func(h: Array) -> Array:
        """Computes value of composed generic class K function.

        Args:
            h (Array): CBF value

        Returns:
            Array: value of composed alpha(h)
        """
        return alpha(h)

    return func
