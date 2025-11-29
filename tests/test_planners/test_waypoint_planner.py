"""
Test Module for cbfkit.planners.waypoint
=========================

This module contains unit tests for waypoint planners.
"""

import unittest

import jax.numpy as jnp
from jax import random

import cbfkit.planners as planners
from cbfkit.utils.user_types import PlannerData

KEY = random.PRNGKey(0)


class TestWaypointPlanner(unittest.TestCase):
    """Tests for the vanilla waypoint planner."""

    def test_vanilla_waypoint(self):
        """Tests that vanilla_waypoint returns the expected x_traj."""
        target_state = jnp.array([1.0, 2.0, 3.0])

        # Create planner
        planner = planners.vanilla_waypoint(target_state=target_state)

        # Execute planner
        # t, x, u_nom (not used), key, data
        t = 0.0
        x = jnp.zeros(3)
        # PlannerCallable = Callable[[float, State, Optional[Control], Key, PlannerData], PlannerCallableReturns]

        # The waypoint_generator defines:
        # def process(t: float, x: State, u_nom: Control, key: Key, data: list)

        # Note: u_nom type is Control, but planner usually takes optional control or just state.
        # user_types says: PlannerCallable = Callable[[float, State, PlannerData], ...]?
        # Let's check user_types.py again to be sure about PlannerCallable signature.

        u_result, data_result = planner(t, x, None, KEY, PlannerData())

        # Verify u_result is usually a zero array or similar, let's check implementation returns u_out = jnp.zeros(x.shape)
        # The implementation returns u_out, new_planner_data
        # u_out is jnp.zeros(x.shape)
        self.assertTrue(jnp.allclose(u_result, jnp.zeros_like(x)))

        # Verify x_traj in data
        self.assertIsNotNone(data_result.x_traj)
        expected_traj = target_state.reshape(-1, 1)
        self.assertTrue(jnp.allclose(data_result.x_traj, expected_traj))


if __name__ == "__main__":
    unittest.main()
