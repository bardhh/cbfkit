"""
#! MANUALLY POPULATE (docstring)
"""
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array, lax
from typing import List, Callable
from cbfkit.certificates import certificate_package

N = 2


###############################################################################
# CLF
###############################################################################


def clf(radius: float, **kwargs) -> Callable[[Array], Array]:
    """Super-level set convention.

    Args:
        #! kwargs -- optional to manually populate

    Returns:
        ret (float): value of goal function evaluated at time and state

    """

    @jit
    def func(state_and_time: Array) -> Array:
        """Function to be evaluated.

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: clf value
        """
        x = state_and_time
        return x[0] ** 2 + x[1] ** 2 - radius

    return func


def clf_grad(radius: float, **kwargs) -> Callable[[Array], Array]:
    """Jacobian for the goal function defined by clf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    jacobian = jacfwd(clf(radius, **kwargs))

    @jit
    def func(state_and_time: Array) -> Array:
        """_summary_

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: clf jacobian (gradient)
        """

        return jacobian(state_and_time)

    return func


def clf_hess(radius: float, **kwargs) -> Callable[[Array], Array]:
    """Hessian for the goal function defined by clf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    hessian = jacrev(jacfwd(clf(radius, **kwargs)))

    @jit
    def func(state_and_time: Array) -> Array:
        """_summary_

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: clf hessian
        """

        return hessian(state_and_time)

    return func


###############################################################################
# CLF1
###############################################################################
clf1_package = certificate_package(clf, clf_grad, clf_hess, N)
