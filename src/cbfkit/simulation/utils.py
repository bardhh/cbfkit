from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array, random

from cbfkit.utils.user_types import (
    Control,
    Covariance,
    Estimate,
    NominalControllerCallable,
    PlannerData,
    State,
    Time,
)


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


def resolve_nominal_control(
    t: Time,
    z: State,
    dt: float,
    key: Array,
    g: Array,
    nominal_controller: Optional[NominalControllerCallable],
    planner_data: PlannerData,
    u_planner: Array,
    has_planner: bool,
) -> Tuple[Array, Array]:
    """Determine the nominal control input from planner output.

    Shared by both eager (backend.py) and JIT (simulator_jit.py) paths.

    Priority:
        1. Planner provided a control trajectory (u_traj) -> use directly.
        2. Planner provided a state trajectory (x_traj) -> track via nominal_controller.
        3. No trajectory -> call nominal_controller with no reference, or zero.

    Returns:
        (u_nom, updated_key)
    """
    if has_planner and planner_data.u_traj is not None:
        return u_planner, key

    if planner_data.x_traj is not None:
        idx = jnp.round(t / dt).astype(int)
        idx = jnp.clip(idx, 0, planner_data.x_traj.shape[1] - 1)
        x_des = planner_data.x_traj[:, idx]
        key, nom_key = random.split(key)
        u_nom, _ = nominal_controller(t, z, nom_key, x_des)
        return u_nom, key

    if nominal_controller is not None:
        key, nom_key = random.split(key)
        u_nom, _ = nominal_controller(t, z, nom_key, None)
        return u_nom, key

    return jnp.zeros((g.shape[1],)), key
