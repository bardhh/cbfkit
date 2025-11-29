from typing import Any, List, NamedTuple, Optional, Union

from jax import Array

from cbfkit.utils.user_types import Control, Covariance, Estimate, State


class SimulationStepData(NamedTuple):
    """Represents the data captured at a single simulation step."""

    state: State
    control: Control
    estimate: Estimate
    covariance: Covariance
    controller_keys: List[str]
    controller_values: List[Any]
    planner_keys: List[str]
    planner_values: List[Any]
