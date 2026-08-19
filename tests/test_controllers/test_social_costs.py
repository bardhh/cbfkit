"""Social-navigation cost terms for MPPI (``cbfkit.controllers.mppi.social_costs``)."""

import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.controllers.mppi.social_costs import (
    SocialCostWeights,
    asymmetric_gaussian,
    pack_state,
    social_cost_terms,
    social_trajectory_cost,
    time_to_collision,
)


def test_asymmetric_gaussian_is_wider_in_front_than_behind_and_grows_with_speed():
    heading = jnp.array([1.0, 0.0])  # pedestrian walks +x
    front = asymmetric_gaussian(jnp.array([1.0, 0.0]), heading, speed=1.0)
    behind = asymmetric_gaussian(jnp.array([-1.0, 0.0]), heading, speed=1.0)
    side = asymmetric_gaussian(jnp.array([0.0, 1.0]), heading, speed=1.0)
    assert front > side > behind
    slow = asymmetric_gaussian(jnp.array([1.0, 0.0]), heading, speed=0.3)
    assert front > slow  # a faster pedestrian needs more room in front
    still = asymmetric_gaussian(jnp.array([1.0, 0.0]), jnp.zeros(2), speed=0.0)
    still_b = asymmetric_gaussian(jnp.array([-1.0, 0.0]), jnp.zeros(2), speed=0.0)
    assert float(still) == pytest.approx(float(still_b))  # standing: isotropic


def test_time_to_collision_matches_the_analytic_value():
    rel = jnp.array([3.0, 0.0])  # pedestrian 3 m ahead ...
    v_rel = jnp.array([-1.0, 0.0])  # ... closing at 1 m/s
    assert float(time_to_collision(rel, v_rel, radius=0.5)) == pytest.approx(2.5)
    assert float(time_to_collision(rel, jnp.array([1.0, 0.0]), radius=0.5)) == jnp.inf  # receding
    assert float(time_to_collision(rel, jnp.array([0.0, 1.0]), radius=0.5)) == jnp.inf  # misses
    assert float(time_to_collision(jnp.array([0.2, 0.0]), v_rel, radius=0.5)) == 0.0  # overlap


def _crossing_scenario(robot_offset_y, robot_vx, n_steps=25, dt=0.2):
    """Pedestrian walks +y through x = 2 at 1 m/s; robot walks +x along y = robot_offset_y."""
    states = []
    for k in range(n_steps):
        t = k * dt
        robot = jnp.array([robot_vx * t, robot_offset_y, robot_vx, 0.0])
        ped = jnp.array([[2.0, -2.0 + 1.0 * t, 0.0, 1.0]])
        states.append(pack_state(robot, ped))
    return jnp.stack(states, axis=1)  # (dim, H)


def test_cost_prefers_passing_behind_a_crossing_pedestrian_over_cutting_in_front():
    w = SocialCostWeights()
    goal = jnp.array([10.0, 0.0])
    # Robot at 0.5 m/s reaches x = 2 at t = 4 s, when the pedestrian is at (2, 2) heading +y:
    # along y = 0 the robot crosses 2 m *behind* her, along y = 4 it crosses 2 m *in front*.
    behind = _crossing_scenario(robot_offset_y=0.0, robot_vx=0.5)
    in_front = _crossing_scenario(robot_offset_y=4.0, robot_vx=0.5)
    u = jnp.zeros((2, behind.shape[1]))
    cost = social_trajectory_cost(n_agents=1, goal=goal, weights=w, dt=0.2)
    tb = social_cost_terms(behind, u, n_agents=1, goal=goal, weights=w, dt=0.2)
    tf = social_cost_terms(in_front, u, n_agents=1, goal=goal, weights=w, dt=0.2)
    assert float(tf["proxemics"]) > float(tb["proxemics"])
    # The goal term is equal (same x progress); the social terms decide.
    assert float(cost(0.0, in_front, u, None)) > float(cost(0.0, behind, u, None))


def test_waiting_is_cheaper_than_cutting_in_front_when_the_gap_is_short():
    """Pedestrian crosses x = 1 walking +y at 1.2 m/s from y = -1.5 (she is at x = 1, y = 0 at
    t = 1.25 s). Robot at the origin heading +x: at 0.8 m/s it walks straight into her path
    (cut), at 0.5 m/s it passes 0.9 m behind her (go), at 0 m/s it waits 1 m from her path."""
    w = SocialCostWeights()
    goal = jnp.array([10.0, 0.0])
    dt, H = 0.2, 25

    def traj(vx):
        s = []
        for k in range(H):
            t = k * dt
            robot = jnp.array([vx * t, 0.0, vx, 0.0])
            ped = jnp.array([[1.0, -1.5 + 1.2 * t, 0.0, 1.2]])
            s.append(pack_state(robot, ped))
        return jnp.stack(s, axis=1)

    cut, go, wait = traj(0.8), traj(0.5), traj(0.0)
    u = jnp.zeros((2, H))
    cost = social_trajectory_cost(n_agents=1, goal=goal, weights=w, dt=dt)
    c_cut, c_go, c_wait = (float(cost(0.0, s, u, None)) for s in (cut, go, wait))
    assert c_cut > c_wait  # standing still beats walking into her path
    assert c_go < c_cut  # passing behind beats cutting in front
    terms = social_cost_terms(wait, u, n_agents=1, goal=goal, weights=w, dt=dt)
    assert float(terms["goal"]) > 0.0  # waiting is not free: the progress term is what we pay


def test_pass_side_penalises_the_wrong_side_only_when_oncoming():
    w = SocialCostWeights(pass_side=1.0)
    goal = jnp.array([10.0, 0.0])
    H, dt = 5, 0.2

    def traj(ped_y, ped_vx):
        s = []
        for k in range(H):
            t = k * dt
            robot = jnp.array([0.5 * t, 0.0, 0.5, 0.0])
            ped = jnp.array([[2.0 + ped_vx * t, ped_y, ped_vx, 0.0]])
            s.append(pack_state(robot, ped))
        return jnp.stack(s, axis=1)

    u = jnp.zeros((2, H))
    kw = dict(n_agents=1, goal=goal, weights=w, dt=dt)
    left = social_cost_terms(traj(0.6, -1.0), u, pass_side="left", **kw)["pass_side"]
    right = social_cost_terms(traj(-0.6, -1.0), u, pass_side="left", **kw)["pass_side"]
    # "left" convention: keep left, i.e. the oncoming pedestrian should pass on the robot's RIGHT
    # (negative y here). A pedestrian on the robot's left (y > 0) is on the wrong side.
    assert float(left) > float(right)
    same_dir = social_cost_terms(traj(0.6, +1.0), u, pass_side="left", **kw)["pass_side"]
    assert float(same_dir) < float(left)  # not oncoming: (almost) no side preference


def test_terms_are_finite_with_zero_agents_and_many_agents():
    goal = jnp.array([5.0, 5.0])
    w = SocialCostWeights(pass_side=0.5)
    for n in (0, 7):
        rng = np.random.default_rng(n)
        H = 6
        robot = jnp.array([0.0, 0.0, 0.3, 0.3])
        peds = jnp.asarray(rng.normal(size=(n, 4)))
        states = jnp.stack([pack_state(robot, peds)] * H, axis=1)
        u = jnp.asarray(rng.normal(size=(2, H)))
        cost = social_trajectory_cost(n_agents=n, goal=goal, weights=w, dt=0.2, pass_side="left")
        assert jnp.isfinite(cost(0.0, states, u, None))
        terms = social_cost_terms(
            states, u, n_agents=n, goal=goal, weights=w, dt=0.2, pass_side="left"
        )
        assert all(jnp.isfinite(v) for v in terms.values())
