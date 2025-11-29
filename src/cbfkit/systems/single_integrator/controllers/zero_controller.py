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


def zero_controller(dynamics=None, **kwargs) -> NominalControllerCallable:
    @jit
    def controller(
        t: float, x: State, key: Key, xd: Optional[State] = None
    ) -> ControllerCallableReturns:
        return jnp.zeros(2), ControllerData()

    return controller
