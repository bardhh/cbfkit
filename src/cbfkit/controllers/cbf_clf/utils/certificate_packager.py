"""
certificate_packager.py
================

This module contains functions to package certificate functions and their gradient vectors.

Functions
---------
-certificate_package: packages the certificate function and its gradients
-concatenate_certificates: combines multiple certificate packages into usable form

Notes
-----
The functions in this module are important to the definition and use of CBFs
and CLFs for control.

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
>>> package1 = certificate_package(cbf, cbf_grad, cbf_hess, n=1)
>>> package2 = certificate_package(cbf, cbf_grad, cbf_hess, n=1)
>>>
>>> limit = 1.0
>>> alpha = 1.0
>>> barriers = concatenate_certificates(
>>>     package1(certificate_conditions=linear_class_k(alpha), limit=limit),
>>>     package2(certificate_conditions=linear_class_k(alpha), limit=-limit),
>>> )

"""

from typing import Any, Callable, Dict, List, Tuple

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import (
    CertificateCallable,
    CertificateCollection,
    CertificateConditionsCallable,
    CertificateHessianCallable,
    CertificateJacobianCallable,
    CertificatePartialCallable,
)


def certificate_package(
    func: Callable[..., Callable[[Array], Array]],
    func_grad: Callable[..., Callable[[Array], Array]],
    func_hess: Callable[..., Callable[[Array], Array]],
    n: int,
) -> Callable[..., CertificateCollection]:
    """Function for packaging and later creating CBF executables.

    Args:
        func (Callable): certificate function
        func_grad (Callable): certificate gradient vector function
        func_hess (Callable): certificate hessian matrix function
        n (int): state dimension

    Returns:
        CertificateCollection: _description_
    """

    def package(
        certificate_conditions: CertificateConditionsCallable,
        **kwargs: Dict[str, Any],
    ) -> CertificateCollection:
        """_summary_

        Args:
            certificate_conditions (Callable): inequality conditions for certificate function
            kwargs


        Returns:
            BarrierTuple: _description_
        """
        v_func = func(**kwargs)
        j_func = func_grad(**kwargs)
        h_func = func_hess(**kwargs)
        t_func = func_grad(**kwargs)
        c_func = certificate_conditions

        @jit
        def v_(t: float, x: Array) -> Array:
            return v_func(jnp.hstack([x, t]))

        @jit
        def j_(t: float, x: Array) -> Array:
            return j_func(jnp.hstack([x, t]))[:n]

        @jit
        def h_(t: float, x: Array) -> Array:
            return h_func(jnp.hstack([x, t]))[:n, :n]

        @jit
        def t_(t: float, x: Array) -> Array:
            return t_func(jnp.hstack([x, t]))[-1]

        return (
            [v_],
            [j_],
            [h_],
            [t_],
            [c_func],
        )

    return package


def concatenate_certificates(*tuples: CertificateCollection) -> CertificateCollection:
    """_summary_

    Returns:
        _type_: _description_
    """
    if not tuples:
        return ([], [], [], [], [])

    # Initialize empty lists for each component of CertificateCollection
    v_list = []
    j_list = []
    h_list = []
    t_list = []
    c_list = []

    # Iterate through the tuples and extend the lists
    for tup in tuples:
        v_list.extend(tup[0])
        j_list.extend(tup[1])
        h_list.extend(tup[2])
        t_list.extend(tup[3])
        c_list.extend(tup[4])

    return (v_list, j_list, h_list, t_list, c_list)
