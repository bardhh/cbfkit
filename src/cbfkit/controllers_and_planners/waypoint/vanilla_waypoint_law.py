"""
vanilla_single_waypoint_cplanner.py
================

To give a constant waypoint for navigation

Functions
---------
-vanilla_waypoint: generates the function to compute the control solution to the CBF-CLF-QP

Notes
-----
Relies on the waypoint_generator function defined in waypoint_generator.py
in the containing folder.

Examples
--------
>>> import jax.numpy as jnp
>>> import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
>>> import cbfkit.controllers_and_planners.waypoint as single_waypoint_planner
>>> target_setpoint = single_waypoint_planner.vanilla_waypoint(target_state=goal)
"""

from .waypoint_generator import (
    waypoint_generator,
)

vanilla_waypoint = waypoint_generator()
