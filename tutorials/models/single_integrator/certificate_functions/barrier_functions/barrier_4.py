"""
#! MANUALLY POPULATE (docstring)
"""
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array, lax
from typing import List, Callable
from cbfkit.certificates import certificate_package

N = 12


###############################################################################
# CBF
###############################################################################


def cbf(r: float, **kwargs) -> Callable[[Array], Array]:
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
        return (x[0] - x[8]) ** 2 + (x[1] - x[9]) ** 2 - r**2

    return func


def cbf_grad(r: float, **kwargs) -> Callable[[Array], Array]:
    """Jacobian for the constraint function defined by cbf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    jacobian = jacfwd(cbf(r, **kwargs))

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


def cbf_hess(r: float, **kwargs) -> Callable[[Array], Array]:
    """Hessian for the constraint function defined by cbf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    hessian = jacrev(jacfwd(cbf(r, **kwargs)))

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
# CBF4
###############################################################################
cbf4_package = certificate_package(cbf, cbf_grad, cbf_hess, N)
