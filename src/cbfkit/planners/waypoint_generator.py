"""
waypoint_generator.py
================

Generates the function for generating CBF-CLF-QP control laws of various forms.

Functions
---------
-waypoint_generator: produces the generating function based on

Notes
-----
Used in the generation of waypoint planner/controller laws.

Examples
--------
>>> import cbfkit.planners as single_waypoint_planner
>>> target_setpoint = single_waypoint_planner.vanilla_waypoint(target_state=goal)
"""

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
    """Function for producing a generating function for MPPI laws of various forms.

    Args:

    Returns:
        (WaypointGenerator): function for generating single waypoint planner law
    """

    def generate_waypoint(
        target_state: Array,
        **kwargs: Dict[str, Any],
    ) -> PlannerCallable:
        """Produces the function to deploy a waypoint enforcer.

        Args:
            **kwargs (Dict[str, Any]): keyword arguments, e.g., RiskAwareParams for RA-CBF-CLF-QP

        Returns:
            PlannerCallable: function for computing control input based on CBF-CLF-QP
        """
        complete = False

        # TODO: define State, Control Trajectory types??
        def process(
            t: Time, x: State, u_nom: Optional[Control], key: Key, data: PlannerData
        ) -> PlannerCallableReturns:
            """Waypoint law.

            Args:
                t (float): time (in sec)
                x (State): state vector

            Returns:
                PlannerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """

            return jittable_process(t, x, key, data)

        @jit
        def jittable_process(
            t: Time, x: State, key: Key, data: PlannerData
        ) -> PlannerCallableReturns:
            """JIT-compatible portion of the Waypoint planner/control law.

            Args:
                t (float): time (in sec)
                x (State): state vector
                u (Array): previous control input trajectory

            Returns:
                PlannerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """
            nonlocal complete

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
