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
>>> from cbfkit.controllers.model_based.cbf_clf_controllers.utils.rectify_relative_degree import rectify_relative_degree
>>>
>>> def f(x):
>>>     return jnp.array([x[2], x[3], x[4], x[5], 0, 0])
>>>
>>> def g(_x):
>>>     return jnp.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])
>>>
>>> def dynamics(x):
>>>     return f(x), g(x), 1
>>>
>>> def constraint(r1, r2):
>>>     def h(x):
>>>         return 1 - x[0] ** 2 - x[1] ** 2 - r1 - r2
>>>     return h
>>>
>>> r1, r2 = 0.1, 0.05
>>>
>>> cbf = rectify_relative_degree(constraint(r1, r2), dynamics, 6)
>>> print(cbf(jnp.array([0.5, 0.5, -0.1, 0, 0.2, 0])))
"""

from typing import Callable, Union, List
from jax import random, jacfwd, jacrev, Array, jit
import jax.numpy as jnp
import jaxlib
import numpy as np
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
)
from cbfkit.utils.user_types import DynamicsCallable, CertificateCollection

# For random sample generation
KEY = random.PRNGKey(0)
KEY, SUBKEY = random.split(KEY)


def kwargs_wrapper(func: Callable) -> Callable:
    def wrapper(**kwargs):
        return func

    return wrapper


def rectify_relative_degree(
    function: Callable[[float, Array], Array],
    system_dynamics: DynamicsCallable,
    state_dim: int,
    roots: Union[Array, None] = None,
    form: str = "exponential",
) -> CertificateCollection:
    """Rectifies the relative degree of the provided constraint function with respect to
    the system dynamics deriving a new exponential- or high-order-CBF.

    Args:
        function (Callable[[float, Array], Array]): constraint function
        system_dynamics (DynamicsCallable): dynamics function
        state_dim (int): dimension of the state
        form (str, optional): type of cascading procedure. Defaults to "exponential".

    Returns:
        Callable[[Dict[str, Any]], CertificateTuple]: Function to create the CertificateCollection
    """
    function_list = compute_function_list(function, system_dynamics, state_dim + 1, form)

    if form == "exponential":
        n_fl = len(function_list)
        if roots is None:
            # Root locations in the left half-plane
            roots = jnp.array([-0.1 * (ii + 1) for ii in range(n_fl)])

        n_r = len(roots)
        if n_r > n_fl:
            roots = roots[:n_fl]
        elif n_r < n_fl:
            roots = jnp.hstack([roots, roots[-1] * jnp.ones((n_fl - n_r))])

        assert jnp.all(roots < 0), "All roots must be in open left-half plane!"
        assert len(roots) == n_fl, "Length of roots must be relative-degree of system minus 1!"

        # Calculate the polynomial coefficients using JAX and SciPy
        polynomial_coefficients = polynomial_coefficients_from_roots(roots)

        def cbf(**kwargs):

            @jit
            def func(x: Array) -> Array:
                return jnp.sum(
                    jnp.array(
                        [
                            func(x) * coeff
                            for func, coeff in zip(function_list, polynomial_coefficients)
                        ]
                    )
                )

            return func

        def cbf_grad(**kwargs) -> Callable[[Array], Array]:
            jacobian = jacfwd(cbf(**kwargs))

            @jit
            def func(x: Array) -> Array:
                return jacobian(x)

            return func

        def cbf_hess(**kwargs) -> Callable[[Array], Array]:
            hessian = jacfwd(jacrev(cbf(**kwargs)))

            @jit
            def func(x: Array) -> Array:
                return hessian(x)

            return func

    elif form == "high-order":

        def cbf(x: Array) -> Array:
            return function_list[0](x)

    return certificate_package(cbf, cbf_grad, cbf_hess, state_dim)


def compute_function_list(
    function: Callable[[float, Array], Array],
    system_dynamics: DynamicsCallable,
    state_dim: int,
    form: str = "exponential",
    func_list: Union[List[Callable[[float, Array], Array]], None] = None,
    subkey: Union[jaxlib.xla_extension.ArrayImpl, None] = None,
    n_samples: int = 10,
):
    """Computes the cascading list of derivatives/functions for rectifying the relative
    degree of the provided function.

    Args:
        function (Callable[[float, Array], Array]): constraint function
        system_dynamics (DynamicsCallable): dynamics function
        state_dim (int): state dimension
        form (str, optional): cascading procedure name. Defaults to "exponential".
        func_list (Union[List[Callable[[float, Array], Array]], None], optional): cascaded list of functions. Defaults to None.
        subkey (Union[jaxlib.xla_extension.ArrayImpl, None], optional): random subkey. Defaults to None.
        n_samples (int, optional): number of samples used to determine relative-degree. Defaults to 10.

    Returns:
        List[Callable]: list of functions/derivatives
    """
    if func_list is None:
        func_list = []

    func_list.append(function)

    if subkey is None:
        subkey = SUBKEY
    else:
        _, subkey = random.split(KEY)

    # Do this at every level
    samples = random.normal(subkey, (n_samples, state_dim))
    jacobian = jacfwd(function)

    total = 0
    for sample in samples:
        _, dyn_g = system_dynamics(sample[:-1])
        grad = jacobian(sample)[:-1]
        total += jnp.sum(jnp.abs(jnp.matmul(grad, dyn_g)))

    def exponential_new_func(x: Array):
        return jnp.matmul(jacobian(x)[:-1], system_dynamics(x[:-1])[0])

    def highorder_new_func(x: Array):
        return jnp.matmul(jacobian(x)[:-1], system_dynamics(x[:-1])[0]) + function(x)

    if total == 0:
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
            subkey,
        )

    return func_list


def polynomial_coefficients_from_roots(roots: Array) -> Array:
    """Computes the nth-order polynomial (real) coefficients given n roots.

    Args:
        roots (Array): roots of polynomial

    Returns:
        Array: polynomial coefficients
    """

    # Create a polynomial with roots at the specified locations
    polynomial = np.poly1d(roots, r=True)

    # Get the coefficients of the polynomial
    coefficients = jnp.array(polynomial.coeffs)

    return coefficients
