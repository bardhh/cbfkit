from typing import Any, Callable, List, Union

import jax.numpy as jnp
import jaxlib
import numpy as np
from jax import Array, jacfwd, jacrev, jit, random

from cbfkit.certificates import certificate_package
from cbfkit.certificates.rectifiers import polynomial_coefficients_from_roots
from cbfkit.utils.user_types import CertificateCollection, DynamicsCallable

# For random sample generation
KEY = random.PRNGKey(0)
KEY, SUBKEY = random.split(KEY)


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
) -> Callable[..., CertificateCollection]:
    """Rectifies the relative degree of the provided constraint function with respect to the system
    dynamics deriving a new exponential- or high-order-CBF.

    Args:
        function (Callable[[Array], Array]): constraint function (takes combined state+time Array)
        system_dynamics (DynamicsCallable): dynamics function
        state_dim (int): dimension of the state
        form (str, optional): type of cascading procedure. Defaults to "exponential".

    Returns
    -------
        Callable[[Dict[str, Any]], CertificateCollection]: Function to create the CertificateCollection
    """
    function_list = compute_function_list(function, system_dynamics, state_dim + 1, form)

    # These will be the callables passed to certificate_package
    cbf_final: Callable[..., Callable[[Array], Array]]
    cbf_grad_final: Callable[..., Callable[[Array], Array]]
    cbf_hess_final: Callable[..., Callable[[Array], Array]]

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

        def cbf_exp(**kwargs) -> Callable[[Array], Array]:
            @jit
            def func(x_and_t: Array) -> Array:
                # x_and_t combines state and time.
                return jnp.sum(
                    jnp.array(
                        [
                            f(x_and_t) * coeff
                            for f, coeff in zip(function_list, polynomial_coefficients)
                        ]
                    )
                )

            return func

        def cbf_grad_exp(**kwargs) -> Callable[[Array], Array]:
            def func_wrapper(
                x_and_t: Array,
            ) -> (
                Array
            ):  # Wrapper needed for jacfwd to get correct Callable[[Array],Array] type for certificate_package
                return cbf_exp(**kwargs)(x_and_t)

            jacobian = jacfwd(func_wrapper)

            @jit
            def func(x_and_t: Array) -> Array:
                return jacobian(x_and_t)

            return func

        def cbf_hess_exp(**kwargs) -> Callable[[Array], Array]:
            def func_wrapper(x_and_t: Array) -> Array:
                return cbf_exp(**kwargs)(x_and_t)

            hessian = jacfwd(jacrev(func_wrapper))

            @jit
            def func(x_and_t: Array) -> Array:
                return hessian(x_and_t)

            return func

        cbf_final = cbf_exp
        cbf_grad_final = cbf_grad_exp
        cbf_hess_final = cbf_hess_exp

    elif form == "high-order":

        def cbf_high(**kwargs) -> Callable[[Array], Array]:
            @jit
            def func(x_and_t: Array) -> Array:
                return function_list[0](x_and_t)

            return func

        def cbf_grad_high(**kwargs) -> Callable[[Array], Array]:
            def func_wrapper(x_and_t: Array) -> Array:
                return cbf_high(**kwargs)(x_and_t)

            jacobian = jacfwd(func_wrapper)

            @jit
            def func(x_and_t: Array) -> Array:
                return jacobian(x_and_t)

            return func

        def cbf_hess_high(**kwargs) -> Callable[[Array], Array]:
            def func_wrapper(x_and_t: Array) -> Array:
                return cbf_high(**kwargs)(x_and_t)

            hessian = jacfwd(jacrev(func_wrapper))

            @jit
            def func(x_and_t: Array) -> Array:
                return hessian(x_and_t)

            return func

        cbf_final = cbf_high
        cbf_grad_final = cbf_grad_high
        cbf_hess_final = cbf_hess_high

    else:  # Fallback to high-order if form is unknown/unspecified

        def cbf_fallback(**kwargs) -> Callable[[Array], Array]:
            @jit
            def func(x_and_t: Array) -> Array:
                return function_list[0](x_and_t)

            return func

        def cbf_grad_fallback(**kwargs) -> Callable[[Array], Array]:
            def func_wrapper(x_and_t: Array) -> Array:
                return cbf_fallback(**kwargs)(x_and_t)

            jacobian = jacfwd(func_wrapper)

            @jit
            def func(x_and_t: Array) -> Array:
                return jacobian(x_and_t)

            return func

        def cbf_hess_fallback(**kwargs) -> Callable[[Array], Array]:
            def func_wrapper(x_and_t: Array) -> Array:
                return cbf_fallback(**kwargs)(x_and_t)

            hessian = jacfwd(jacrev(func_wrapper))

            @jit
            def func(x_and_t: Array) -> Array:
                return hessian(x_and_t)

            return func

        cbf_final = cbf_fallback
        cbf_grad_final = cbf_grad_fallback
        cbf_hess_final = cbf_hess_fallback

    return certificate_package(cbf_final, cbf_grad_final, cbf_hess_final, state_dim)


def compute_function_list(
    function: Callable[[Array], Array],
    system_dynamics: DynamicsCallable,
    state_dim: int,  # state_dim is now combined state+time dimension
    form: str = "exponential",
    func_list: Union[List[Callable[[Array], Array]], None] = None,
    _subkey: Union[jaxlib.xla_extension.ArrayImpl, None] = None,
    n_samples: int = 10,
):
    """Computes the cascading list of derivatives/functions for rectifying the relative degree of
    the provided function.

    Args:
        function (Callable[[Array], Array]): constraint function (takes combined state+time Array)
        system_dynamics (DynamicsCallable): dynamics function
        state_dim (int): state dimension (combined state+time)
        form (str, optional): cascading procedure name. Defaults to "exponential".
        func_list (Union[List[Callable[[Array], Array]], None], optional): cascaded list of functions. Defaults to None.
        n_samples (int, optional): number of samples used to determine relative-degree. Defaults to 10.

    Returns
    -------
        List[Callable]: list of functions/derivatives
    """
    if func_list is None:
        func_list = []

    func_list.append(function)

    # Deterministically probe a small set of points to detect non-zero Lie derivatives.
    jacobian = jacfwd(function)
    probe_points = jnp.vstack(
        [
            jnp.zeros((1, state_dim)),
            jnp.eye(state_dim)[: n_samples - 1, :],
        ]
    )

    total = jnp.array(0.0)
    for sample in probe_points:
        _, dyn_g = system_dynamics(sample[:-1])
        grad = jacobian(sample)[:-1]
        total += jnp.sum(jnp.abs(jnp.matmul(grad, dyn_g)))

    def exponential_new_func(x_and_t: Array):
        f_val, _ = system_dynamics(x_and_t[:-1])
        return jnp.matmul(jacobian(x_and_t)[:-1], f_val)

    def highorder_new_func(x_and_t: Array):
        f_val, _ = system_dynamics(x_and_t[:-1])
        return jnp.matmul(jacobian(x_and_t)[:-1], f_val) + function(x_and_t)

    if total <= 1e-9:
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
            None,
            n_samples,
        )

    return func_list
