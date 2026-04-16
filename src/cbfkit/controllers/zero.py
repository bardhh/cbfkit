"""Zero controller factory.

Creates a nominal controller that always returns zero control inputs.
Used as a default when no nominal controller is needed.
"""

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import (
    ControllerCallableReturns,
    ControllerData,
    Key,
    NominalControllerCallable,
    Optional,
    State,
)


def zero_controller(n_controls: int = 2, **kwargs) -> NominalControllerCallable:
    """Create a nominal controller that returns zero control inputs.

    Args:
        n_controls: Number of control inputs.
        **kwargs: Ignored. For API compatibility with system-specific factories.

    Returns:
        A controller callable that returns ``jnp.zeros(n_controls)``.
    """

    @jit
    def controller(
        t: float, x: State, key: Key, xd: Optional[State] = None
    ) -> ControllerCallableReturns:
        return jnp.zeros(n_controls), ControllerData()

    return controller
