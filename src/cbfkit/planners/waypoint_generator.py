"""Functions for generating waypoint laws."""

from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import (
    Control,
    Key,
    PlannerCallable,
    PlannerCallableReturns,
    PlannerData,
    State,
    Time,
)


def waypoint_generator() -> Callable[..., PlannerCallable]:
    """Return a factory that builds simple waypoint planner laws."""

    def generate_waypoint(
        target_state: Array,
        **kwargs: Dict[str, Any],
    ) -> PlannerCallable:
        """Build a planner that holds the target waypoint in planner metadata."""

        def process(
            t: Time, x: State, u_nom: Optional[Control], key: Key, data: PlannerData
        ) -> PlannerCallableReturns:
            """Wrapper around the jittable waypoint update."""
            return jittable_process(t, x, key, data)

        @jit
        def jittable_process(
            t: Time, x: State, key: Key, data: PlannerData
        ) -> PlannerCallableReturns:
            """Update planner state for the waypoint and output zero nominal control."""
            # If no control input is explicitly defined, set to zeros
            u_out = jnp.zeros(x.shape)  # Placeholder for actual control logic

            # Update PlannerData
            new_planner_data = data._replace(
                x_traj=target_state.reshape(-1, 1),
                u_traj=None,  # u_traj would be for a planned trajectory of controls, here it's a single waypoint
            )

            return u_out, new_planner_data

        return process

    return generate_waypoint
