"""
certificate_packager.py
================

This module contains functions to package certificate functions and their gradient vectors.

Functions
---------
-certificate_package: packages the certificate function and its gradients. Supports automated
 differentiation if gradients/Hessians are not provided.
-generate_certificate: simpler interface to create a CertificateCollection from a function.
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
                user_grad = func_grad(**kwargs)
            else:
                user_grad = func_grad  # type: ignore[assignment]

            # Aegis: Wrap manual gradient to accept concatenated input 'xt'
            if input_style == CertificateInputStyle.STATE:
                # User provided grad(x). We need j_func(xt) -> [grad(x), 0]
                def j_func(xt):
                    gx = user_grad(xt[:-1])
                    gx = jnp.atleast_1d(gx)
                    return jnp.append(gx, 0.0)

                t_func = j_func

            elif input_style == CertificateInputStyle.SEPARATED:
                # User provided grad(t, x). We need j_func(xt)
                # Aegis: Robustly handle manual gradients for separated input style.
                # If user returns spatial gradient (dx, len=n), auto-append temporal gradient (dt).
                # If user returns full gradient (len=n+1), use as is.

                # Capture _orig_v_sep for temporal partial derivative (defined above)
                grad_t_auto = grad(_orig_v_sep, argnums=0)

                def j_func(xt):
                    t = xt[-1]
                    x = xt[:-1]
                    gx = user_grad(t, x)
                    gx = jnp.atleast_1d(gx)

                    # Check if user returned full gradient or spatial gradient
                    if gx.shape[0] == n:
                        # Spatial gradient provided. Append auto-diff temporal gradient.
                        gt = grad_t_auto(t, x)
                        return jnp.append(gx, gt)
                    elif gx.shape[0] == n + 1:
                        # Full gradient provided (legacy behavior).
                        return gx
                    else:
                        raise ValueError(
                            f"Manual gradient shape mismatch for SEPARATED input style. "
                            f"Expected shape ({n},) (spatial only) or ({n+1},) (full [dx, dt]), "
                            f"but got {gx.shape}."
                        )

                t_func = j_func

            else:
                # User provided grad(xt)
                j_func = user_grad
                t_func = user_grad

        if func_hess is None:
            # Auto-differentiate
            h_func = hessian(v_func)
        else:
            if use_factory:
                user_hess = func_hess(**kwargs)
            else:
                user_hess = func_hess  # type: ignore[assignment]

            # Aegis: Wrap manual hessian to accept concatenated input 'xt'
            if input_style == CertificateInputStyle.STATE:
                # User provided hess(x). We need h_func(xt)
                def h_func(xt):
                    return user_hess(xt[:-1])

            elif input_style == CertificateInputStyle.SEPARATED:
                # User provided hess(t, x). We need h_func(xt)
                def h_func(xt):
                    return user_hess(xt[-1], xt[:-1])

            else:
                h_func = user_hess

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


def generate_certificate(
    certificate: Callable[..., Array],
    certificate_conditions: CertificateConditionsCallable,
    input_style: Union[
        Literal["state", "separated"], CertificateInputStyle
    ] = CertificateInputStyle.STATE,
    certificate_grad: Optional[Callable[..., Array]] = None,
    certificate_hess: Optional[Callable[..., Array]] = None,
) -> CertificateCollection:
    """Creates a CertificateCollection from a single function, automatically computing derivatives.

    Unlike `certificate_package`, this function does not use the factory pattern and does not
    require specifying the state dimension.

    Args:
        certificate: The certificate function h(x) or h(t, x).
        certificate_conditions: Function defining the conditions (e.g. Class K).
        input_style: Signature of the certificate function.
            - "state" (default): certificate(x)
            - "separated": certificate(t, x)
        certificate_grad: Optional manual gradient function. If provided, must match input_style.
        certificate_hess: Optional manual hessian function. If provided, must match input_style.

    Returns:
        CertificateCollection: A collection containing the function and its derivatives.
    """
    # Normalize input_style to enum
    if isinstance(input_style, str):
        try:
            input_style = CertificateInputStyle(input_style)
        except ValueError:
            valid_styles = ["state", "separated"]
            raise ValueError(
                f"Invalid input_style '{input_style}' for generate_certificate. Must be one of {valid_styles}"
            ) from None

    if input_style == CertificateInputStyle.CONCATENATED:
        raise ValueError(
            "generate_certificate does not support 'concatenated' input style. "
            "Use 'state' or 'separated', or use `certificate_package` instead."
        )

    # Wrap the certificate function to always accept (t, x)
    if input_style == CertificateInputStyle.STATE:
        # User provided h(x)
        def v_func_canonical(t: float, x: Array) -> Array:
            return certificate(x)

        # Gradient w.r.t x
        if certificate_grad is None:
            grad_x = grad(certificate)

            def j_func_canonical(t: float, x: Array) -> Array:
                return grad_x(x)
        else:
            # Manual grad h(x)
            def j_func_canonical(t: float, x: Array) -> Array:
                return certificate_grad(x)  # type: ignore

        # Hessian w.r.t x
        if certificate_hess is None:
            hess_x = hessian(certificate)

            def h_func_canonical(t: float, x: Array) -> Array:
                return hess_x(x)
        else:
            # Manual hess h(x)
            def h_func_canonical(t: float, x: Array) -> Array:
                return certificate_hess(x)  # type: ignore

        # Partial w.r.t t (always 0 for state-only function)
        def t_func_canonical(t: float, x: Array) -> Array:
            return jnp.array(0.0)

    else:
        # User provided h(t, x)
        def v_func_canonical(t: float, x: Array) -> Array:
            return certificate(t, x)

        # Gradient w.r.t x (arg 1)
        if certificate_grad is None:
            grad_x = grad(certificate, argnums=1)

            def j_func_canonical(t: float, x: Array) -> Array:
                return grad_x(t, x)
        else:
            # Manual grad h(t, x) w.r.t x?
            # Convention: if manual grad is provided for h(t, x), it should return grad_x h(t, x)
            # Or should it return full gradient? Let's assume it matches the auto-diff output: dh/dx.
            def j_func_canonical(t: float, x: Array) -> Array:
                return certificate_grad(t, x)  # type: ignore

        # Hessian w.r.t x (arg 1)
        if certificate_hess is None:
            hess_x = hessian(certificate, argnums=1)

            def h_func_canonical(t: float, x: Array) -> Array:
                return hess_x(t, x)
        else:
            def h_func_canonical(t: float, x: Array) -> Array:
                return certificate_hess(t, x)  # type: ignore

        # Partial w.r.t t (arg 0)
        # Note: If user provides manual grad, we still need partial t.
        # Currently, if manual grad is provided, we assume partial t is handled separately or user only cared about x?
        # CertificateCollection needs partials.
        # If manual grad is provided, we can't infer partial t unless user provides it?
        # For now, let's auto-diff partial t unless we add a `certificate_partial` argument.
        # Let's keep it simple: always auto-diff partial t for now, as it's rarely manually optimized.
        grad_t = grad(certificate, argnums=0)

        def t_func_canonical(t: float, x: Array) -> Array:
            return grad_t(t, x)

    # JIT compile the canonical functions
    v_jit = jit(v_func_canonical)
    j_jit = jit(j_func_canonical)
    h_jit = jit(h_func_canonical)
    t_jit = jit(t_func_canonical)

    return CertificateCollection(
        [v_jit],
        [j_jit],
        [h_jit],
        [t_jit],
        [certificate_conditions],
    )


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
