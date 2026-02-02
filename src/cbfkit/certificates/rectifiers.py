"""
rectify_relative_degree.py
================

This module contains the function responsible for rectifying the relative-degree of
a constraint function given a system dynamics model, i.e., for returning a new
CertificateCollection object with a barrier function of relative-degree one.

Functions
---------
-rectify_relative_degree: rectifies the relative-degree of the provided constraint function
-compute_function_list: computes the cascading list of derivatives/functions for rectify_relative_degree
-polynomial_coefficients_from_roots: computes the nth order polynomial coefficients given n roots

Notes
-----
The rectify_relative_degree function is useful for modifying some arbitrary constraint function
for use as a CBF in a QP-based control law. If the constraint function is already of relative-degree
one with respect to the system dynamics, then the constraint function is not modified.

Example
-------
>>> from cbfkit.certificates import rectify_relative_degree
>>> from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
>>>
>>> def f(x):
>>>     return jnp.array([x[2], x[3], x[4], x[5], 0, 0])
>>>
>>> def g(_x):
>>>     return jnp.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])
>>>
>>> def dynamics(x):
>>>     return f(x), g(x)
>>>
>>> def constraint(r1, r2):
>>>     def h(x):
>>>         return 1 - x[0] ** 2 - x[1] ** 2 - r1 - r2
>>>     return h
>>>
>>> r1, r2 = 0.1, 0.05
>>>
>>> # New simpler usage
>>> cbf = rectify_relative_degree(
>>>     constraint(r1, r2),
>>>     dynamics,
>>>     6,
>>>     certificate_conditions=zeroing_barriers.linear_class_k(1.0)
>>> )
"""

from typing import Callable, List, Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import Array, jacfwd, jacrev, jit, random

from cbfkit.certificates import certificate_package
from cbfkit.utils.user_types import (
    CertificateCollection,
    CertificateConditionsCallable,
    DynamicsCallable,
)

# For random sample generation
KEY = random.PRNGKey(0)
KEY, SUBKEY = random.split(KEY)

# Tolerance for determining if relative degree is reached.
# If total absolute control authority over random samples is below this threshold,
# we treat it as zero (effective relative degree is higher).
RELATIVE_DEGREE_TOLERANCE = 1e-6


def kwargs_wrapper(func: Callable) -> Callable:
    def wrapper(**kwargs):
        return func

    return wrapper


def rectify_relative_degree(
    function: Callable[[Array], Array],
    system_dynamics: DynamicsCallable,
    state_dim: int,
    roots: Union[Array, None] = None,
    form: str = "exponential",
    certificate_conditions: Optional[CertificateConditionsCallable] = None,
    rng: Optional[Union[int, Array]] = None,
) -> Union[Callable[..., CertificateCollection], CertificateCollection]:
    """Rectifies the relative degree of the provided constraint function with respect to the system
    dynamics deriving a new exponential- or high-order-CBF.

    Args:
        function (Callable[[float, Array], Array]): constraint function
        system_dynamics (DynamicsCallable): dynamics function
        state_dim (int): dimension of the state
        form (str, optional): type of cascading procedure. Defaults to "exponential".
        certificate_conditions (Optional[CertificateConditionsCallable]): certificate conditions.
            If provided, returns a CertificateCollection directly. Defaults to None.
        rng (Optional[Union[int, Array]]): random seed or key for relative degree sampling.
            Defaults to None (using global default).

    Returns
    -------
        Union[Callable[..., CertificateCollection], CertificateCollection]: Function to create the CertificateCollection,
            or the CertificateCollection itself if certificate_conditions is provided.
    """
    subkey: Array
    if rng is None:
        subkey = SUBKEY  # type: ignore[assignment]
    elif isinstance(rng, int):
        subkey = random.PRNGKey(rng)  # type: ignore[assignment]
    else:
        subkey = rng  # type: ignore[assignment]

    function_list = compute_function_list(
        function, system_dynamics, state_dim + 1, form, subkey=subkey
    )

    if form == "exponential":
        n_fl = len(function_list)
        n_roots = n_fl - 1
        if roots is None:
            # Root locations in the left half-plane
            roots = jnp.array([-0.1 * (ii + 1) for ii in range(n_roots)])

        n_r = len(roots)
        if n_r > n_roots:
            roots = roots[:n_roots]
        elif n_r < n_roots:
            roots = jnp.hstack([roots, roots[-1] * jnp.ones((n_roots - n_r))])

        assert jnp.all(roots < 0), "All roots must be in open left-half plane!"
        assert len(roots) == n_roots, "Length of roots must be relative-degree of system minus 1!"

        # Calculate the polynomial coefficients using JAX and SciPy
        polynomial_coefficients = polynomial_coefficients_from_roots(roots)

        def cbf():
            @jit
            def func(x: Array) -> Array:
                return jnp.sum(
                    jnp.array(
                        [
                            func(x) * coeff
                            for func, coeff in zip(function_list, polynomial_coefficients[::-1])
                        ]
                    )
                )

            return func

    elif form == "high-order":

        def cbf():
            def func(x: Array) -> Array:
                return function_list[-1](x)

            return func

    factory = certificate_package(cbf, n=state_dim)

    if certificate_conditions is not None:
        return factory(certificate_conditions)

    return factory


def compute_function_list(
    function: Callable[[Array], Array],
    system_dynamics: DynamicsCallable,
    state_dim: int,
    form: str = "exponential",
    func_list: Union[List[Callable[[Array], Array]], None] = None,
    subkey: Union[Array, None] = None,
    n_samples: int = 10,
):
    """Computes the cascading list of derivatives/functions for rectifying the relative degree of
    the provided function.

    Args:
        function (Callable[[float, Array], Array]): constraint function
        system_dynamics (DynamicsCallable): dynamics function
        state_dim (int): state dimension
        form (str, optional): cascading procedure name. Defaults to "exponential".
        func_list (Union[List[Callable[[float, Array], Array]], None], optional): cascaded list of functions. Defaults to None.
        subkey (Union[jaxlib.xla_extension.ArrayImpl, None], optional): random subkey. Defaults to None.
        n_samples (int, optional): number of samples used to determine relative-degree. Defaults to 10.

    Returns
    -------
        List[Callable]: list of functions/derivatives
    """
    if func_list is None:
        func_list = []

    func_list.append(function)

    if subkey is None:
        subkey = SUBKEY  # type: ignore[assignment]

    # Split the key to ensure different samples are used at each recursion level
    sample_key, next_subkey = random.split(subkey)  # type: ignore[arg-type]

    # Do this at every level
    samples = random.normal(sample_key, (n_samples, state_dim))  # type: ignore[arg-type]
    jacobian = jacfwd(function)

    total = jnp.array(0.0)
    for sample in samples:
        _, dyn_g = system_dynamics(sample[:-1])
        grad = jacobian(sample)[:-1]
        total += jnp.sum(jnp.abs(jnp.matmul(grad, dyn_g)))

    def exponential_new_func(x: Array):
        return jnp.matmul(jacobian(x)[:-1], system_dynamics(x[:-1])[0]) + jacobian(x)[-1]

    def highorder_new_func(x: Array):
        return (
            jnp.matmul(jacobian(x)[:-1], system_dynamics(x[:-1])[0])
            + jacobian(x)[-1]
            + function(x)
        )

    if jnp.isnan(total):
        raise ValueError(
            "Encountered NaN during relative degree verification. Check your dynamics function and constraint function for numerical stability."
        )

    if total < RELATIVE_DEGREE_TOLERANCE:
        if form == "exponential":
            new_func = exponential_new_func

        elif form == "high-order":
            new_func = highorder_new_func

        else:
            new_func = highorder_new_func

        return compute_function_list(
            new_func,
            system_dynamics,
            state_dim,
            form,
            func_list,
            next_subkey,
        )

    return func_list


def polynomial_coefficients_from_roots(roots: Array) -> Array:
    """Computes the nth-order polynomial (real) coefficients given n roots.

    Args:
        roots (Array): roots of polynomial

    Returns
    -------
        Array: polynomial coefficients
    """
    # Create a polynomial with roots at the specified locations
    polynomial = np.poly1d(roots, r=True)

    # Get the coefficients of the polynomial
    coefficients = jnp.array(polynomial.coeffs)

    return coefficients
