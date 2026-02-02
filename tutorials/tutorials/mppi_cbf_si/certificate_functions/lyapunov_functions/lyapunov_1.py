"""
Lyapunov function 1
"""
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array, lax
from typing import Callable
from cbfkit.certificates import certificate_package

N = 2


###############################################################################
# CLF
###############################################################################


def clf(goal: Array, **kwargs) -> Callable[[Array], Array]:
    """Super-level set convention.

    Args:
        goal (Array): goal state

    Returns:
        ret (Array): value of goal function evaluated at time and state

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
        return (x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2

    return func


def clf_grad(goal: Array, **kwargs) -> Callable[[Array], Array]:
    """Jacobian for the goal function defined by clf.

    Args:
        goal (Array): goal state

    Returns:
        ret (Array): value of goal function evaluated at time and state

    """
    jacobian = jacfwd(clf(goal, **kwargs))

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


def clf_hess(goal: Array, **kwargs) -> Callable[[Array], Array]:
    """Hessian for the goal function defined by clf.

    Args:
        goal (Array): goal state

    Returns:
        ret (Array): value of goal function evaluated at time and state

    """
    hessian = jacrev(jacfwd(clf(goal, **kwargs)))

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
