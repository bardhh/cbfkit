"""
#! MANUALLY POPULATE (docstring)
"""
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array, lax
from typing import List, Callable
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
)

N = 8


###############################################################################
# CBF
###############################################################################


def cbf(xo: float, yo: float, a: float, b: float, r: float, **kwargs) -> Callable[[Array], Array]:
    """Super-level set convention.

    Args:
        #! kwargs -- optional to manually populate

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """

    @jit
    def func(state_and_time: Array) -> Array:
        """Function to be evaluated.

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: cbf value
        """
        x = state_and_time
        return ((x[0] - xo) / a) ** 10 + ((x[1] - yo) / b) ** 10 - 1 - r

    return func


def cbf_grad(
    xo: float, yo: float, a: float, b: float, r: float, **kwargs
) -> Callable[[Array], Array]:
    """Jacobian for the constraint function defined by cbf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    jacobian = jacfwd(cbf(xo, yo, a, b, r, **kwargs))

    @jit
    def func(state_and_time: Array) -> Array:
        """_summary_

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: cbf jacobian (gradient)
        """

        return jacobian(state_and_time)

    return func


def cbf_hess(
    xo: float, yo: float, a: float, b: float, r: float, **kwargs
) -> Callable[[Array], Array]:
    """Hessian for the constraint function defined by cbf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    hessian = jacrev(jacfwd(cbf(xo, yo, a, b, r, **kwargs)))

    @jit
    def func(state_and_time: Array) -> Array:
        """_summary_

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: cbf hessian
        """

        return hessian(state_and_time)

    return func


###############################################################################
# CBF1
###############################################################################
cbf1_package = certificate_package(cbf, cbf_grad, cbf_hess, N)
