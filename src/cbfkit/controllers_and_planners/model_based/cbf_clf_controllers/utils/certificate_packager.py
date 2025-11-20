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

import jax.numpy as jnp
from jax import Array, jit
from typing import Dict, Any, Callable
from cbfkit.utils.user_types import (
    CertificateCallable,
    CertificateJacobianCallable,
    CertificateHessianCallable,
    CertificatePartialCallable,
    CertificateConditionsCallable,
    CertificateTuple,
)


def certificate_package(
    func: Callable[[Dict[str, Any]], Callable[[Array], Array]],
    func_grad: Callable[[Dict[str, Any]], Callable[[Array], Array]],
    func_hess: Callable[[Dict[str, Any]], Callable[[Array], Array]],
    n: int,
) -> Callable[[Dict[str, Any]], CertificateTuple]:
    """Function for packaging and later creating CBF executables.

    Args:
        func (Callable): certificate function
        func_grad (Callable): certificate gradient vector function
        func_hess (Callable): certificate hessian matrix function
        n (int): state dimension

    Returns:
        CertificateTuple: _description_
    """

    def package(
        certificate_conditions: CertificateConditionsCallable,
        **kwargs: Dict[str, Any],
    ) -> CertificateTuple:
        """_summary_

        Args:
            certificate_conditions (Callable): inequality conditions for certificate function
            kwargs


        Returns:
            BarrierTuple: _description_
        """
        v_func: CertificateCallable = func(**kwargs)
        j_func: CertificateJacobianCallable = func_grad(**kwargs)
        h_func: CertificateHessianCallable = func_hess(**kwargs)
        t_func: CertificatePartialCallable = func_grad(**kwargs)
        c_func: CertificateConditionsCallable = certificate_conditions

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


def concatenate_certificates(*tuples: CertificateTuple):
    """_summary_

    Returns:
        _type_: _description_
    """
    # Initialize an empty tuple to store the result
    result_tuple = ()

    # Iterate through the tuples and concatenate the lists
    for tup in zip(*tuples):
        # Use list comprehension to concatenate corresponding lists
        result_list = [x for sublist in tup for x in sublist]
        # Append the result_list to the result_tuple
        result_tuple += (result_list,)

    return result_tuple
