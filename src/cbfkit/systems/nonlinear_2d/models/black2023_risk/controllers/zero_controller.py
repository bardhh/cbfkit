"""Zero controller for nonlinear 2D black2023_risk (2 control inputs)."""

from typing import Callable, Optional, Tuple

from jax import Array

from cbfkit.controllers.zero import zero_controller as _zero
from cbfkit.utils.user_types import ControllerData, Time


def zero_controller() -> (
    Callable[[Time, Array, Array, Optional[Array]], Tuple[Array, ControllerData]]
):
    return _zero(n_controls=2)
