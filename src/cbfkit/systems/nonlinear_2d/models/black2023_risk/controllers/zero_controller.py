"""Zero controller for nonlinear 2D black2023_risk (2 control inputs)."""

from cbfkit.controllers.zero import zero_controller as _zero
from cbfkit.utils.user_types import ControllerCallable


def zero_controller() -> ControllerCallable:
    return _zero(n_controls=2)
