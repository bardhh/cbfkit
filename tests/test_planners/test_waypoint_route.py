"""waypoint_route: stateless waypoint sequencing through PlannerData.x_traj."""

import jax.numpy as jnp
from jax import random

from cbfkit.planners import waypoint_route
from cbfkit.utils.user_types import PlannerData

KEY = random.PRNGKey(0)
WPS = jnp.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])


def _wp(data):
    return data.x_traj[:, 0]


def test_starts_at_first_waypoint_and_holds_until_within_radius():
    planner = waypoint_route(WPS, radius=0.2)
    _, d = planner(0.0, jnp.array([0.0, 0.0, 9.0]), None, KEY, PlannerData())
    assert jnp.allclose(_wp(d), WPS[0])
    _, d = planner(0.1, jnp.array([0.7, 0.0, 9.0]), None, KEY, d)  # 0.3 away: hold
    assert jnp.allclose(_wp(d), WPS[0])


def test_advances_when_reached_and_clamps_at_last():
    planner = waypoint_route(WPS, radius=0.2)
    d = PlannerData.from_constant(WPS[0])
    _, d = planner(0.0, jnp.array([0.95, 0.0]), None, KEY, d)  # within 0.2 -> next
    assert jnp.allclose(_wp(d), WPS[1])
    _, d = planner(0.1, jnp.array([2.0, 1.0]), None, KEY, d)
    assert jnp.allclose(_wp(d), WPS[2])
    _, d = planner(0.2, jnp.array([3.0, 0.0]), None, KEY, d)  # last reached: stays
    assert jnp.allclose(_wp(d), WPS[2])
    assert d.u_traj is None


def test_position_indices_select_the_planar_coordinates():
    planner = waypoint_route(WPS, radius=0.2, position_indices=(2, 3))
    d = PlannerData.from_constant(WPS[0])
    _, d = planner(0.0, jnp.array([9.0, 9.0, 1.0, 0.0]), None, KEY, d)
    assert jnp.allclose(_wp(d), WPS[1])
