import jax.numpy as jnp
from jax import jit, Array, scipy
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns


def fxt_lyapunov_controller(epsilon: float) -> ControllerCallable:
    """
    Create a zero controller for the given fixed-wing uav dynamics.

    Args:
        None

    Returns:
        controller (Callable): handle to function computing zero control

    """

    @jit
    def controller(_t: float, x: Array) -> ControllerCallableReturns:
        """Computes Lyapunov function-based control input (1x1).

        Args:
            _t (float): time in sec
            x (Array): state vector (or estimate if using observer/estimator)

        Returns:
            u (Array): 1x1 zero vector
            data: (dict): empty dictionary
        """
        V = x[0] ** 2 + x[1] ** 2
        fV = -((V) ** 0.5) - (V) ** 1.5
        ux2 = (fV + 2 * x[0] * x[1]) / 2 + epsilon * (1 - x[0] ** 2) * x[1] ** 2 - x[0] * x[1]
        # logging data
        data = {}

        return jnp.array([ux2]), data

    return controller


def fxt_stochastic_lyapunov_controller(
    epsilon: float, sigma_sum_squares: float
) -> ControllerCallable:
    """
    Create a zero controller for the given fixed-wing uav dynamics.

    Args:
        None

    Returns:
        controller (Callable): handle to function computing zero control

    """

    @jit
    def controller(_t: float, x: Array) -> ControllerCallableReturns:
        """Computes Lyapunov function-based control input (1x1).

        Args:
            _t (float): time in sec
            x (Array): state vector (or estimate if using observer/estimator)

        Returns:
            u (Array): 1x1 zero vector
            data: (dict): empty dictionary
        """
        V = x[0] ** 2 + x[1] ** 2
        fV = -((V) ** 0.5) - (V) ** 1.5
        ux2 = (
            (fV - sigma_sum_squares + 2 * x[0] * x[1]) / 2
            + epsilon * (1 - x[0] ** 2) * x[1] ** 2
            - x[0] * x[1]
        )
        # logging data
        data = {}

        return jnp.array([ux2]), data

    return controller


def fxt_risk_aware_lyapunov_controller(
    epsilon: float,
    sigma_sum_squares: float,
    pg: float,
    t_reach: float,
    vartheta: float,
) -> ControllerCallable:
    """
    Create a zero controller for the given fixed-wing uav dynamics.

    Args:
        None

    Returns:
        controller (Callable): handle to function computing zero control

    """
    a = 100.0
    b = 1.0
    r = jnp.sqrt(2 * t_reach) * vartheta * scipy.special.erfinv(2 * pg - 1)

    # @jit
    def controller(_t: float, x: Array) -> ControllerCallableReturns:
        """Computes Lyapunov function-based control input (1x1).

        Args:
            _t (float): time in sec
            x (Array): state vector (or estimate if using observer/estimator)

        Returns:
            u (Array): 1x1 zero vector
            data: (dict): empty dictionary
        """
        V = r + a * x[0] ** 2 + b * x[1] ** 2
        fV = -((V) ** 0.5) - (V) ** 1.5
        ux2 = (
            (fV - sigma_sum_squares + 2 * a * x[0] * x[1]) / (2 * b)
            + epsilon * (1 - x[0] ** 2) * x[1] ** 2
            - x[0] * x[1]
        )
        # logging data
        data = {}

        return jnp.array([ux2]), data

    return controller
