import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
)

#! Circular Obstacle Avoidance
N = 3


def clf(goal: Array, radius: float) -> Array:
    """Reach goal convergence function.

    Args:
        goal (Array): [x, y] location of goal
        radius (float): goal radius

    Returns:
        Callable[[Array], Array]: callable clf
    """

    @jit
    def func(state_and_time: Array) -> Array:
        """ """
        x_e, y_e, _theta_e, _t = state_and_time
        x_o, y_o, _ = goal

        return (x_e - x_o) ** 2 + (y_e - y_o) ** 2 - radius**2

    return func


def clf_grad(goal: Array, radius: float) -> Array:
    """Jacobian for Reach goal convergence function.

    Args:
        goal (Array): [x, y] location of goal
        radius (float): goal radius

    Returns:
        Callable[[Array], Array]: callable clf
    """
    jacobian = jacfwd(clf(goal, radius))

    @jit
    def func(state_and_time: Array) -> Array:
        """ """
        return jacobian(state_and_time)

    return func


def clf_hess(goal: Array, radius: float) -> Array:
    """Hessian for Reach goal convergence function.

    Args:
        goal (Array): [x, y] location of goal
        radius (float): goal radius

    Returns:
        Callable[[Array], Array]: callable clf
    """
    hessian = jacrev(jacfwd(clf(goal, radius)))

    @jit
    def func(state_and_time: Array) -> Array:
        """ """
        return hessian(state_and_time)

    return func


###############################################################################
# 2nd Order CBF
###############################################################################
reach_goal = certificate_package(clf, clf_grad, clf_hess, N)
