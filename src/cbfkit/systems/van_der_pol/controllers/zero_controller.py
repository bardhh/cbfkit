"""Zero controller for van der Pol (1 control input)."""

from cbfkit.controllers.zero import zero_controller as _zero
from cbfkit.utils.user_types import NominalControllerCallable


def zero_controller() -> NominalControllerCallable:
    return _zero(n_controls=1)
