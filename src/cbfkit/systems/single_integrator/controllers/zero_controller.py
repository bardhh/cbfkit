"""Zero controller for single integrator (2 control inputs)."""

from cbfkit.controllers.zero import zero_controller as _zero
from cbfkit.utils.user_types import NominalControllerCallable


def zero_controller(dynamics=None, **kwargs) -> NominalControllerCallable:
    return _zero(n_controls=2)
