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
>>> import cbfkit.controllers_and_planners.waypoint as single_waypoint_planner
>>> target_setpoint = single_waypoint_planner.vanilla_waypoint(target_state=goal)
"""

from typing import Union, Dict, Any, Optional
import jax.numpy as jnp
from jax import Array, lax, jit

from cbfkit.utils.user_types import (
    ControllerCallable,
    ControllerCallableReturns,
    State,
    Control,
    Key,
)


def waypoint_generator(
) -> float:
    """Function for producing a generating function for MPPI laws of various forms.

    Args:

    Returns:
        (WaypointGenerator): function for generating single waypoint planner law
    """

    def generate_waypoint(
        target_state: Array,
        **kwargs: Dict[str, Any],
    ) -> ControllerCallable:
        """Produces the function to deploy a waypoint enforcer.

        Args:
            **kwargs (Dict[str, Any]): keyword arguments, e.g., RiskAwareParams for RA-CBF-CLF-QP

        Returns:
            ControllerCallable: function for computing control input based on CBF-CLF-QP
        """
        complete = False

        # TODO: define State, Control Trajectory types??
        def process(
            t: float, x: State, u_nom: Control, key: Key, data: list
        ) -> ControllerCallableReturns:
            """Waypoint law.

            Args:
                t (float): time (in sec)
                x (State): state vector

            Returns:
                ControllerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """

            return jittable_process(t, x, key, data)

        @jit
        def jittable_process(t: float, x: State, key: Key, data: list) -> ControllerCallableReturns:
            """JIT-compatible portion of the Waypoint planner/control law.

            Args:
                t (float): time (in sec)
                x (State): state vector
                u (Array): previous control input trajectory

            Returns:
                ControllerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """
            nonlocal complete

            return None, {"u_traj": None, "x_traj": target_state.reshape(-1, 1)}

        return process

    return generate_waypoint
