import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import (
    Control,
    ControllerCallableReturns,
    ControllerData,
    Key,
    NominalControllerCallable,
    Optional,
    State,
)


def zero_controller() -> NominalControllerCallable:
    """
    Create a zero controller for the given fixed-wing uav dynamics.

    Args:
        None

    Returns:
        controller (Callable): handle to function computing zero control

    """

    @jit
    def controller(
        _t: float, _state: State, _key: Key, _xd: Optional[State] = None
    ) -> ControllerCallableReturns:
        """Computes zero control input (1x1).

        Args:
            _t (float): time in sec
            _state (Array): state vector (or estimate if using observer/estimator)
            _key (Key): unused
            _xd (Optional[State]): unused

        Returns:
            zeros (Array): 1x1 zero vector
            data: ControllerData: empty ControllerData
        """
        # logging data
        data = ControllerData()

        return jnp.zeros((1,)), data

    return controller
