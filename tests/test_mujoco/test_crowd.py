"""SocialForceCrowd: goal-seeking pedestrians that yield to the robot, each other and pillars."""

import jax.numpy as jnp
import pytest

pytest.importorskip("mujoco")

from cbfkit.systems.mujoco.crowd import SocialForceCrowd


def test_initial_state_heads_for_the_goal_at_the_desired_speed():
    crowd = SocialForceCrowd(starts=[(0.0, 0.0)], goals=[(3.0, 4.0)], speeds=[0.5])
    assert crowd.n == 1
    assert jnp.allclose(crowd.x0[0], jnp.array([0.0, 0.0, 0.3, 0.4]))


def test_pedestrian_yields_to_a_robot_in_its_way_and_to_pillars():
    crowd = SocialForceCrowd(
        starts=[(0.0, 0.0)], goals=[(10.0, 0.0)], speeds=[0.5], obstacles=[(1.0, -0.3)]
    )
    s = crowd.x0
    free = SocialForceCrowd(starts=[(0.0, 0.0)], goals=[(10.0, 0.0)], speeds=[0.5]).step(
        0.0, jnp.array([50.0, 50.0]), s, 0.1
    )
    assert jnp.allclose(free[0, 2:], s[0, 2:], atol=1e-6)  # nothing nearby: keeps walking
    blocked = crowd.step(0.0, jnp.array([0.8, 0.1]), s, 0.1)  # robot just ahead, pillar ahead-right
    assert float(blocked[0, 2]) < float(s[0, 2])  # slows down
    assert float(blocked[0, 3]) > 0.0  # and sidesteps away from robot (+y) / pillar (at -y)
    assert float(blocked[0, 0]) > 0.0  # position integrated


def test_pedestrians_repel_each_other():
    crowd = SocialForceCrowd(
        starts=[(0.0, 0.0), (0.9, 0.0)], goals=[(10.0, 0.0), (-10.0, 0.0)], speeds=[0.5, 0.5]
    )
    nxt = crowd.step(0.0, jnp.array([50.0, 50.0]), crowd.x0, 0.1)
    assert float(nxt[0, 2]) < 0.5 and float(nxt[1, 2]) > -0.5  # both slow down head-on
