"""
certificate_packager.py
================

This module contains functions to package certificate functions and their gradient vectors.

Functions
---------
-certificate_package: packages the certificate function and its gradients. Supports automated
 differentiation if gradients/Hessians are not provided.
-concatenate_certificates: combines multiple certificate packages into usable form

Notes
-----
The functions in this module are important to the definition and use of CBFs
and CLFs for control. Returns `CertificateCollection` NamedTuples.

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
>>> # Auto-diff example
>>> package1 = certificate_package(cbf, n=1)
>>>
>>> limit = 1.0
>>> alpha = 1.0
>>> barriers = concatenate_certificates(
>>>     package1(certificate_conditions=linear_class_k(alpha), limit=limit),
>>> )

"""

from typing import Any, Callable, Dict, Literal, Optional, Union

import jax.numpy as jnp
from jax import Array, grad, hessian, jit

from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CertificateCallable,
    CertificateCollection,
    CertificateConditionsCallable,
    CertificateHessianCallable,
    CertificateInputStyle,
    CertificateJacobianCallable,
    CertificatePartialCallable,
)


CertificateInputStyleName = Literal["concatenated", "separated", "state"]


def certificate_package(
    func: Callable[..., Callable[[Array], Array]],
    func_grad: Optional[Callable[..., Callable[[Array], Array]]] = None,
    func_hess: Optional[Callable[..., Callable[[Array], Array]]] = None,
    n: int = 0,
    input_style: Union[
        CertificateInputStyleName, CertificateInputStyle
    ] = CertificateInputStyle.CONCATENATED,
    use_factory: bool = True,
) -> Callable[..., CertificateCollection]:
    """Function for packaging and later creating CBF executables.

    Args:
        func (Callable): certificate function factory (or function if use_factory=False)
        func_grad (Callable, optional): certificate gradient vector function factory (or function).
            If None, computed automatically using jax.grad.
        func_hess (Callable, optional): certificate hessian matrix function factory (or function).
            If None, computed automatically using jax.hessian.
        n (int): state dimension
        input_style (str | CertificateInputStyle): expected signature of the certificate function.
            - "concatenated" (default): func returns f(xt) where xt is [x, t].
            - "separated": func returns f(t, x).
            - "state": func returns f(x).
        use_factory (bool): whether func, func_grad, and func_hess are factories (returning the function)
            or the functions themselves. Defaults to True (factories).

    Returns
    -------
        CertificateCollection: _description_
    """
    # Validate input_style
    if isinstance(input_style, str):
        try:
            input_style = CertificateInputStyle(input_style)
        except ValueError:
            valid_styles = [e.value for e in CertificateInputStyle]
            raise ValueError(
                f"Invalid input_style '{input_style}'. Must be one of {valid_styles}"
            ) from None

    def package(
        certificate_conditions: CertificateConditionsCallable,
        **kwargs: Dict[str, Any],
    ) -> CertificateCollection:
        """_summary_

        Args:
            certificate_conditions (Callable): inequality conditions for certificate function
            kwargs


        Returns
        -------
            BarrierTuple: _description_
        """
        if use_factory:
            v_func = func(**kwargs)
        else:
            v_func = func  # type: ignore[assignment]

        if input_style == CertificateInputStyle.SEPARATED:
            _orig_v_sep = v_func

            def v_func(xt):
                return _orig_v_sep(xt[-1], xt[:-1])

        elif input_style == CertificateInputStyle.STATE:
            _orig_v_state = v_func

            def v_func(xt):
                return _orig_v_state(xt[:-1])

        if func_grad is None:
            # Auto-differentiate
            j_func = grad(v_func)
            t_func = j_func
        else:
            if use_factory:
                j_func = func_grad(**kwargs)
                t_func = func_grad(**kwargs)
            else:
                j_func = func_grad  # type: ignore[assignment]
                t_func = func_grad  # type: ignore[assignment]

        if func_hess is None:
            # Auto-differentiate
            h_func = hessian(v_func)
        else:
            if use_factory:
                h_func = func_hess(**kwargs)
            else:
                h_func = func_hess  # type: ignore[assignment]

        c_func = certificate_conditions

        @jit
        def v_(t: float, x: Array) -> Array:
            return v_func(jnp.hstack([x, t]))

        @jit
        def j_(t: float, x: Array) -> Array:
            # Gradient w.r.t x is the first n elements
            return j_func(jnp.hstack([x, t]))[:n]

        @jit
        def h_(t: float, x: Array) -> Array:
            # Hessian w.r.t x is the top-left nxn block
            return h_func(jnp.hstack([x, t]))[:n, :n]

        @jit
        def t_(t: float, x: Array) -> Array:
            # Partial w.r.t t is the last element of the gradient
            return t_func(jnp.hstack([x, t]))[-1]

        return CertificateCollection(
            [v_],
            [j_],
            [h_],
            [t_],
            [c_func],
        )

    return package


def concatenate_certificates(*tuples: CertificateCollection) -> CertificateCollection:
    """Concatenates multiple CertificateCollections into one.

    Note:
        You can also use the `+` operator to concatenate CertificateCollections directly.

    Args:
        *tuples: Variable number of CertificateCollection objects.

    Returns:
        CertificateCollection: A single collection containing all certificate functions and derivatives.
    """
    if not tuples:
        return EMPTY_CERTIFICATE_COLLECTION

    return sum(tuples, EMPTY_CERTIFICATE_COLLECTION)
