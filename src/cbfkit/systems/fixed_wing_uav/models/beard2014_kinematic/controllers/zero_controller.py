"""Zero controller for fixed-wing UAV (3 control inputs)."""

from cbfkit.controllers.zero import zero_controller as _zero
from cbfkit.utils.user_types import NominalControllerCallable


def zero_controller() -> NominalControllerCallable:
    return _zero(n_controls=3)
