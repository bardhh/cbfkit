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
>>> from cbfkit.certificates import certificate_package, concatenate_certificates
>>> from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
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
    """Generates the class K function alpha(h) for the zeroing CBF condition:

    hdot + alpha(h) >= 0

    which is equivalent to:
    hdot >= -alpha(h)

    where alpha(h) = alpha * h.

    Args:
        alpha (float): The coefficient for the linear class K function.

    Returns
    -------
        Callable[[Array], Array]: The class K function alpha(h).
    """
    assert alpha >= 0
    return lambda h: alpha * h


def cubic_class_k(alpha: float) -> Callable[[Array], Array]:
    """Generates the class K function alpha(h) for the zeroing CBF condition:

    hdot + alpha(h) >= 0

    which is equivalent to:
    hdot >= -alpha(h)

    where alpha(h) = alpha * h^3.

    Args:
        alpha (float): The coefficient for the cubic class K function.

    Returns
    -------
        Callable[[Array], Array]: The class K function alpha(h).
    """
    assert alpha >= 0
    return lambda h: alpha * h**3


def generic_class_k(alpha: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Generates the class K function alpha(h) for the zeroing CBF condition:

    hdot + alpha(h) >= 0

    which is equivalent to:
    hdot >= -alpha(h)

    Args:
        alpha (Callable[[Array], Array]): The class K function itself.

    Returns
    -------
        Callable[[Array], Array]: The class K function alpha(h).
    """

    def func(h: Array) -> Array:
        """Computes value of composed generic class K function.

        Args:
            h (Array): CBF value

        Returns
        -------
            Array: value of composed alpha(h)
        """
        return alpha(h)

    return func
