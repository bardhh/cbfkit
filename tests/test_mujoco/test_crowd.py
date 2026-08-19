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


def test_vectorised_forces_match_the_library_policy():
    """The batched social force equals cbfkit's per-pedestrian ``social_force_policy``."""
    from cbfkit.systems.mujoco.crowd import social_force_accelerations
    from cbfkit.systems.pedestrian.behaviors import social_force_policy

    states = jnp.array([[0.0, 0.0, 0.5, 0.0], [0.9, 0.1, -0.4, 0.0], [0.3, -0.8, 0.0, 0.3]])
    goals = jnp.array([[10.0, 0.0], [-10.0, 0.0], [0.0, 10.0]])
    speeds = jnp.array([0.5, 0.4, 0.3])
    others = jnp.array([[0.5, 0.6], [-0.7, 0.0]])  # robot, pillar
    kw = dict(
        relaxation_time=0.5,
        repulsion_strength=2.0,
        repulsion_range=0.6,
        ped_radius=0.3,
        agent_radius=0.35,
    )
    acc = social_force_accelerations(states, goals, speeds, others, **kw)
    for i in range(3):
        pol = social_force_policy(
            goal=goals[i],
            desired_speed=float(speeds[i]),
            relaxation_time=0.5,
            repulsion_strength=2.0,
            repulsion_range=0.6,
            pedestrian_radius=0.3,
            agent_radius=0.35,
        )
        o = jnp.concatenate([jnp.delete(states[:, :2], i, axis=0), others], axis=0)
        ref = pol(0.0, states[i], {"others_states": o})
        assert jnp.allclose(acc[i], ref, atol=1e-6), (i, acc[i], ref)
