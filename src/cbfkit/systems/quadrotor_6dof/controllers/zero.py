import jax.numpy as jnp
from jax import jit, Array
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns


def zero_controller() -> ControllerCallable:
    """
    Create a zero controller for the given quadrotor dynamics.

    Args:
        None

    Returns:
        controller (Callable): handle to function computing zero control

    """

    @jit
    def controller(_t: float, _state: Array) -> ControllerCallableReturns:
        """Computes zero control input (4x1).

        Args:
            _t (float): time in sec
            _state (Array): state vector (or estimate if using observer/estimator)

        Returns:
            zeros (Array): 4x1 zero vector
            data: (dict): empty dictionary
        """
        # logging data
        data = {}

        return jnp.zeros((4,)), data

    return controller
