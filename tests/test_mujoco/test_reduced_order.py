"""Reduced-order CBF on the plant's CoM: embedded dynamics, barriers, and the safe-locomotion wrapper."""

import urllib.error

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.systems.mujoco.plant import MujocoPlant
from cbfkit.systems.mujoco.reduced_order import (
    com_obstacle_barriers,
    embedded_single_integrator,
    safe_locomotion_controller,
)
from cbfkit.utils.user_types import ControllerData


@pytest.fixture(scope="module")
def g1_plant():
    g1 = pytest.importorskip("cbfkit.systems.mujoco.g1")
    try:
        return MujocoPlant(g1.load_g1()), g1
    except (RuntimeError, urllib.error.URLError) as exc:
        pytest.skip(f"G1 assets unavailable: {exc}")


def test_embedded_single_integrator_shapes(g1_plant):
    plant, _ = g1_plant
    f, g = embedded_single_integrator(plant.state_dim, plant.com_indices)(
        jnp.zeros(plant.state_dim)
    )
    assert f.shape == (plant.state_dim,) and g.shape == (plant.state_dim, 2)
    assert float(g[plant.com_indices[0], 0]) == 1.0 and float(g[plant.com_indices[1], 1]) == 1.0
    assert float(jnp.abs(g).sum()) == 2.0  # nothing else


def test_barrier_is_evaluated_at_the_com_not_the_pelvis(g1_plant):
    plant, g1mod = g1_plant
    g1 = g1mod.G1(plant.mj_model)
    x = g1.x_stand(plant)
    com = x[plant.com_indices[0] : plant.com_indices[0] + 2]
    # Obstacle centred exactly on the CoM -> h = -1 there; centred on the pelvis xy -> h != -1
    # unless CoM == pelvis (it isn't: measured offset at stand).
    b_com = com_obstacle_barriers(plant, [com], [(0.3, 0.3)])
    b_pel = com_obstacle_barriers(plant, [x[0:2]], [(0.3, 0.3)])
    assert float(b_com.functions[0](0.0, x)) == pytest.approx(-1.0)
    assert float(b_pel.functions[0](0.0, x)) != pytest.approx(-1.0)


def test_cbf_filters_command_away_from_obstacle(g1_plant):
    plant, g1mod = g1_plant
    g1 = g1mod.G1(plant.mj_model)
    x = g1.x_stand(plant)
    com = x[plant.com_indices[0] : plant.com_indices[0] + 2]
    obstacle = com + jnp.array([0.6, 0.0])  # just ahead, ellipsoid radius 0.5 -> h = 0.44
    dyn = embedded_single_integrator(plant.state_dim, plant.com_indices)
    barriers = com_obstacle_barriers(plant, [obstacle], [(0.5, 0.5)], class_k_gain=1.0)
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0]), dynamics_func=dyn, barriers=barriers
    )
    v_nom = jnp.array([1.0, 0.0])  # straight into the obstacle
    v_safe, data = cbf_qp(0.0, x, v_nom, jax.random.PRNGKey(0), ControllerData())
    assert not bool(data.error)
    assert float(v_safe[0]) < float(v_nom[0])  # slowed toward the obstacle
    # Constraint check: Lgh v + alpha h >= 0 with h = ((cx-ox)/a)^2 + ((cy-oy)/b)^2 - 1
    d = com - obstacle
    h = float((d[0] / 0.5) ** 2 + (d[1] / 0.5) ** 2 - 1)
    grad = 2 * d / 0.5**2
    assert float(grad @ v_safe) + 1.0 * h >= -1e-6


def test_safe_locomotion_wrapper_composes_and_logs(g1_plant):
    plant, g1mod = g1_plant
    g1 = g1mod.G1(plant.mj_model)
    x = g1.x_stand(plant)
    dyn = embedded_single_integrator(plant.state_dim, plant.com_indices)
    barriers = com_obstacle_barriers(
        plant, [x[plant.com_indices[0] : plant.com_indices[0] + 2] + 5.0], [(0.5, 0.5)]
    )
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0]), dynamics_func=dyn, barriers=barriers
    )
    seen = {}

    def fake_locomotion(t, xx, cmd, key, data):  # stands in for SamplingMpc.as_controller()
        seen["cmd"] = cmd
        sub = dict(data.sub_data or {})
        sub["_loco"] = jnp.asarray(sub.get("_loco", 0.0)) + 1.0  # carry-only state
        return jnp.zeros(plant.nu), data._replace(sub_data=sub)

    ctrl = safe_locomotion_controller(cbf_qp, fake_locomotion)
    u, d1 = ctrl(0.0, x, jnp.array([0.4, 0.1]), jax.random.PRNGKey(0), ControllerData())
    u, d2 = ctrl(0.02, x, jnp.array([0.4, 0.1]), jax.random.PRNGKey(1), d1)
    assert u.shape == (plant.nu,)
    assert jnp.allclose(seen["cmd"], d2.sub_data["v_safe"])
    assert jnp.allclose(d2.sub_data["v_nom"], jnp.array([0.4, 0.1]))
    assert (
        float(d2.sub_data["_loco"]) == 2.0
    )  # carry-only state survived the CBF's sub_data rewrite
    assert not bool(d2.error)


def test_double_integrator_hocbf_brakes_toward_obstacle(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import (
        com_obstacle_hocbfs,
        embedded_double_integrator,
        safe_locomotion_controller_di,
    )

    plant, g1mod = g1_plant
    g1 = g1mod.G1(plant.mj_model)
    x = g1.x_stand(plant)
    com = x[plant.com_indices[0] : plant.com_indices[0] + 2]
    obstacle = com + jnp.array([0.9, 0.0])  # ahead; keep-out radius 0.6 -> h = 1.25
    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices)
    f, g = dyn(jnp.concatenate([x, jnp.array([0.3, 0.1])]))
    assert f.shape == (plant.state_dim + 2,) and g.shape == (plant.state_dim + 2, 2)
    assert float(f[plant.com_indices[0]]) == pytest.approx(0.3)  # d/dt com = v
    barriers = com_obstacle_hocbfs(plant, [obstacle], [(0.6, 0.6)], class_k_gain=1.0)
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([2.0, 2.0]), dynamics_func=dyn, barriers=barriers
    )
    # Already moving fast at the obstacle: nominal wants to hold 0.5 m/s; the HOCBF must brake.
    xa = jnp.concatenate([x, jnp.array([0.5, 0.0])])
    a_safe, data = cbf_qp(0.0, xa, jnp.array([0.0, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert not bool(data.error)
    assert float(a_safe[0]) < -0.05  # decelerating along x
    # Through the wrapper: integrated command shrinks toward the obstacle and is carried in _di_v.
    seen = {}

    def fake_locomotion(t, xx, cmd, key, d):
        seen["cmd"] = cmd
        return jnp.zeros(plant.nu), d

    ctrl = safe_locomotion_controller_di(cbf_qp, fake_locomotion, plant, 0.02, v_max=0.5)
    d = ControllerData(sub_data={"_di_v": jnp.array([0.5, 0.0])})
    _, d1 = ctrl(0.0, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), d)
    assert float(d1.sub_data["_di_v"][0]) < 0.5
    assert jnp.allclose(seen["cmd"], d1.sub_data["v_safe"])
    assert "a_safe" in d1.sub_data


def test_moving_obstacle_barrier_matches_shifted_static_barrier(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import com_moving_obstacle_barriers

    plant, g1mod = g1_plant
    x = g1mod.G1(plant.mj_model).x_stand(plant)
    p0, v, ell = jnp.array([1.0, 0.5]), jnp.array([-0.3, 0.2]), (0.6, 0.6)
    moving = com_moving_obstacle_barriers(plant, [p0], [v], [ell])
    for t in (0.0, 2.5):
        static = com_obstacle_barriers(plant, [p0 + v * t], [ell])
        assert float(moving.functions[0](t, x)) == pytest.approx(float(static.functions[0](0.0, x)))
    # dh/dt = -2 (com - p(t)) . v / r^2  (the packaged partial), checked against finite differences
    t, eps = 1.0, 1e-5
    fd = (float(moving.functions[0](t + eps, x)) - float(moving.functions[0](t - eps, x))) / (
        2 * eps
    )
    assert abs(fd) > 1e-3  # the test is vacuous if the pedestrian motion does not change h
    assert float(moving.partials[0](t, x)) == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_moving_obstacle_hocbf_reacts_to_approaching_pedestrian(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import (
        com_moving_obstacle_hocbfs,
        com_obstacle_hocbfs,
        embedded_double_integrator,
    )

    plant, g1mod = g1_plant
    x = g1mod.G1(plant.mj_model).x_stand(plant)
    com = x[plant.com_indices[0] : plant.com_indices[0] + 2]
    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices)
    xa = jnp.concatenate([x, jnp.zeros(2)])  # robot standing still, zero commanded velocity
    ped0 = com + jnp.array([1.4, 0.0])  # 1.4 m ahead, keep-out 0.65
    still = com_obstacle_hocbfs(plant, [ped0], [(0.65, 0.65)])
    walking = com_moving_obstacle_hocbfs(plant, [ped0], [jnp.array([-0.6, 0.0])], [(0.65, 0.65)])
    kw = dict(control_limits=jnp.array([1.0, 1.0]), dynamics_func=dyn)
    a_still, d1 = vanilla_cbf_clf_qp_controller(barriers=still, **kw)(
        0.0, xa, jnp.zeros(2), jax.random.PRNGKey(0), ControllerData()
    )
    a_walk, d2 = vanilla_cbf_clf_qp_controller(barriers=walking, **kw)(
        0.0, xa, jnp.zeros(2), jax.random.PRNGKey(0), ControllerData()
    )
    assert not bool(d1.error) and not bool(d2.error)
    assert float(jnp.abs(a_still).max()) < 1e-6  # static obstacle, standing still: nothing to do
    assert float(a_walk[0]) < -0.05  # pedestrian closing in: back away (dh/dt term is active)
    # At a later time the pedestrian is elsewhere: h differs from t=0
    h0 = float(walking.functions[0](0.0, xa))
    h1 = float(walking.functions[0](1.0, xa))
    assert h1 < h0


# --------------------------------------------------------------------------- tracked agents
def test_embedded_double_integrator_with_agents_propagates_agent_velocity(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import agent_slice, embedded_double_integrator

    plant, _ = g1_plant
    n_ag = 2
    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices, n_agents=n_ag)
    agents = jnp.array([[1.0, 2.0, 0.3, -0.1], [5.0, 5.0, 0.0, 0.7]])
    xa = jnp.concatenate([jnp.zeros(plant.state_dim), jnp.array([0.2, 0.1]), agents.reshape(-1)])
    f, g = dyn(xa)
    assert f.shape == (plant.state_dim + 2 + 4 * n_ag,) and g.shape == (f.shape[0], 2)
    assert float(f[plant.com_indices[0]]) == pytest.approx(0.2)
    s0, s1 = agent_slice(plant.state_dim, 0), agent_slice(plant.state_dim, 1)
    assert jnp.allclose(f[s0][:2], agents[0, 2:]) and jnp.allclose(f[s0][2:], 0.0)
    assert jnp.allclose(f[s1][:2], agents[1, 2:])
    assert float(jnp.abs(g).sum()) == 2.0  # control enters only the commanded velocity


def test_agent_hocbf_reacts_to_approaching_agent_and_distance_shape(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import (
        com_agent_hocbfs,
        embedded_double_integrator,
    )

    plant, g1mod = g1_plant
    x = g1mod.G1(plant.mj_model).x_stand(plant)
    com = x[plant.com_indices[0] : plant.com_indices[0] + 2]
    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices, n_agents=1)
    barriers = com_agent_hocbfs(plant, 1, [(0.65, 0.65)], shape="distance")
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0]), dynamics_func=dyn, barriers=barriers
    )
    ahead = com + jnp.array([1.4, 0.0])
    for v_agent, expect_brake in ((jnp.zeros(2), False), (jnp.array([-0.6, 0.0]), True)):
        xa = jnp.concatenate([x, jnp.zeros(2), ahead, v_agent])
        a, d = cbf_qp(0.0, xa, jnp.zeros(2), jax.random.PRNGKey(0), ControllerData())
        assert not bool(d.error)
        if expect_brake:
            assert float(a[0]) < -0.05  # agent walking at the standing robot: back away
        else:
            assert float(jnp.abs(a).max()) < 1e-6
    # distance shape: h = |com - p| / r - 1, unit-norm gradient (times 1/r) -- the property
    # that keeps the robust margin from growing with distance.
    from cbfkit.systems.mujoco.reduced_order import com_obstacle_hocbfs

    for far in (1.0, 4.0):
        b = com_obstacle_hocbfs(
            plant, [com + jnp.array([far, 0.0])], [(0.65, 0.65)], shape="distance"
        )
        xa = jnp.concatenate([x, jnp.zeros(2)])
        assert float(b.functions[0](0.0, xa)) == pytest.approx(far / 0.65 - 1.0, rel=1e-6)
        J = b.jacobians[0](0.0, xa)
        assert float(
            jnp.linalg.norm(J[plant.com_indices[0] : plant.com_indices[0] + 2])
        ) == pytest.approx(1 / 0.65, rel=1e-6)


def test_di_wrapper_steps_and_logs_agents(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import (
        com_agent_hocbfs,
        embedded_double_integrator,
        safe_locomotion_controller_di,
    )

    plant, g1mod = g1_plant
    x = g1mod.G1(plant.mj_model).x_stand(plant)

    class Drift:  # minimal agents object: one agent drifting at constant velocity
        x0 = jnp.array([[3.0, 0.0, -0.5, 0.0]])

        def step(self, t, robot_xy, states, dt):
            return states.at[:, :2].add(states[:, 2:] * dt)

    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices, n_agents=1)
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0]),
        dynamics_func=dyn,
        barriers=com_agent_hocbfs(plant, 1, [(0.65, 0.65)]),
    )
    seen = {}

    def fake_loco(t, xx, cmd, key, d):
        seen["cmd"] = cmd
        return jnp.zeros(plant.nu), d

    ctrl = safe_locomotion_controller_di(cbf_qp, fake_loco, plant, 0.02, agents=Drift())
    _, d1 = ctrl(0.0, x, jnp.array([0.3, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert jnp.allclose(d1.sub_data["agents"], Drift.x0)  # the states the QP saw this step
    assert float(d1.sub_data["_agents"][0, 0]) == pytest.approx(
        3.0 - 0.5 * 0.02
    )  # carried, stepped
    _, d2 = ctrl(0.02, x, jnp.array([0.3, 0.0]), jax.random.PRNGKey(0), d1)
    assert jnp.allclose(d2.sub_data["agents"], d1.sub_data["_agents"])


def test_di_wrapper_local_planner_hook_overrides_a_nom_and_carries_state(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import (
        com_obstacle_hocbfs,
        embedded_double_integrator,
        safe_locomotion_controller_di,
    )

    plant, g1mod = g1_plant
    x = g1mod.G1(plant.mj_model).x_stand(plant)
    ci = plant.com_indices
    seen = {}

    def local_planner(t, xa_c, v_nom, key, sub):
        seen["xa_c"] = xa_c
        n = sub.get("_lp_calls")
        n = jnp.zeros((), dtype=jnp.int32) if n is None else n
        return jnp.array([0.25, -0.25]), jnp.array([0.1, 0.0]), {"_lp_calls": n + 1, "lp_flag": 1.0}

    dyn = embedded_double_integrator(plant.state_dim, ci)
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0]),
        dynamics_func=dyn,
        barriers=com_obstacle_hocbfs(plant, [[50.0, 50.0]], [[0.5, 0.5]]),  # far away: inactive
    )

    def fake_loco(t, xx, cmd, key, d):
        return jnp.zeros(plant.nu), d

    ctrl = safe_locomotion_controller_di(
        cbf_qp, fake_loco, plant, 0.02, local_planner=local_planner
    )
    _, d1 = ctrl(0.0, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert seen["xa_c"].shape == (4,)  # [com | v], no agents
    assert jnp.allclose(seen["xa_c"][:2], x[ci[0] : ci[0] + 2])
    assert jnp.allclose(
        d1.sub_data["a_safe"], jnp.array([0.25, -0.25]), atol=1e-3
    )  # planner a_nom, unfiltered
    assert jnp.allclose(d1.sub_data["v_nom"], jnp.array([0.1, 0.0]))  # logged v_nom is the plan's
    assert int(d1.sub_data["_lp_calls"]) == 1 and float(d1.sub_data["lp_flag"]) == 1.0
    _, d2 = ctrl(0.02, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), d1)
    assert int(d2.sub_data["_lp_calls"]) == 2  # carried


def test_mppi_local_planner_replans_on_schedule_and_holds_in_between():
    from cbfkit.controllers.mppi import vanilla_mppi
    from cbfkit.controllers.mppi.social_costs import SocialCostWeights, social_trajectory_cost
    from cbfkit.systems.mujoco.reduced_order import embedded_double_integrator, mppi_local_planner

    n, H, dt_plan = 1, 8, 0.2
    goal = jnp.array([5.0, 0.0])
    mppi = vanilla_mppi(
        control_limits=jnp.array([1.0, 1.0]),
        dynamics_func=embedded_double_integrator(2, (0, 1), n_agents=n),
        trajectory_cost=social_trajectory_cost(
            n_agents=n, goal=goal, weights=SocialCostWeights(), dt=dt_plan
        ),
        mppi_args={
            "robot_state_dim": 4 + 4 * n,
            "robot_control_dim": 2,
            "prediction_horizon": H,
            "num_samples": 64,
            "time_step": dt_plan,
            "use_GPU": False,
            "costs_lambda": 5.0,
            "cost_perturbation": 0.0,
            "control_std": 0.5,
        },
    )
    planner = mppi_local_planner(mppi, n, horizon=H, replan_every=10)
    xa = jnp.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, -1.0, 0.0])  # pedestrian 3 m ahead, coming
    key = jax.random.PRNGKey(0)
    a0, v0, s0 = planner(0.0, xa, jnp.array([0.5, 0.0]), key, {})
    assert a0.shape == (2,) and jnp.all(jnp.isfinite(a0)) and not bool(s0["mppi_error"])
    assert s0["_mppi_u_traj"].shape == (H, 2) and s0["mppi_x_traj"].shape == (4 + 4 * n, H + 1)
    assert float(a0[0]) > 0.0  # the goal is ahead: it starts walking (the pedestrian is 3 m away)
    # the next 9 calls hold the acceleration and the plan, whatever the state/key
    a1, v1, s1 = planner(0.02, xa + 0.1, jnp.zeros(2), jax.random.PRNGKey(9), s0)
    assert jnp.allclose(a1, a0) and jnp.allclose(s1["mppi_x_traj"], s0["mppi_x_traj"])
    s = s1
    for i in range(2, 10):
        _, _, s = planner(i * 0.02, xa, jnp.zeros(2), key, s)
    a10, _, s10 = planner(0.2, xa, jnp.zeros(2), jax.random.PRNGKey(1), s)
    assert int(s10["_mppi_k"]) == 11
    assert not jnp.allclose(
        s10["_mppi_u_traj"], s0["_mppi_u_traj"]
    )  # re-solved (shifted warm start)
    # and it is jit-compatible with a fixed carry structure (the simulator scans it)
    jitted = jax.jit(lambda t, xa, v, k, s: planner(t, xa, v, k, s))
    aj, vj, sj = jitted(0.4, xa, jnp.zeros(2), key, s10)
    assert jnp.all(jnp.isfinite(aj)) and sj["_mppi_u_traj"].shape == (H, 2)


def test_heading_di_dynamics_and_ellipse_barrier_rotation(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import (
        com_agent_ellipse_hocbfs,
        embedded_heading_double_integrator,
        hd_agent_slice,
    )

    plant, g1mod = g1_plant
    sd = plant.state_dim
    dyn = embedded_heading_double_integrator(sd, plant.com_indices, n_agents=1)
    xa = jnp.zeros(sd + 4 + 4)
    xa = xa.at[sd : sd + 2].set(jnp.array([0.3, 0.1]))  # v
    xa = xa.at[sd + 3].set(0.5)  # omega
    sl = hd_agent_slice(sd, 0)
    xa = xa.at[sl.start : sl.stop].set(jnp.array([2.0, 0.0, -1.0, 0.0]))
    f, g = dyn(xa)
    assert float(f[plant.com_indices[0]]) == pytest.approx(0.3)  # com' = v
    assert float(f[sd + 2]) == pytest.approx(0.5)  # theta' = omega
    assert float(f[sl.start]) == pytest.approx(-1.0)  # p_i' = v_i
    assert g.shape == (sd + 8, 3)
    assert float(g[sd + 3, 2]) == pytest.approx(1.0)  # alpha -> omega'

    # The rotating ellipse: pedestrian dead ahead at 0.55 m; facing it (theta=0) presents
    # the narrow longitudinal axis (0.16+0.3=0.46): h > 0. Turned side-on (theta=pi/2)
    # the wide lateral axis (0.28+0.3=0.58) points at it: h < 0. That sign flip is what
    # lets the QP trade rotation against braking. On a fully static scene the rectified
    # psi equals h, so barriers.functions[0] probes the raw geometry.
    barriers = com_agent_ellipse_hocbfs(plant, 1, (0.16, 0.28), ped_radius=0.30)
    x = g1mod.G1(plant.mj_model).x_stand(plant)
    com = x[plant.com_indices[0] : plant.com_indices[0] + 2]

    def h_at(theta):
        z = jnp.concatenate(
            [x, jnp.zeros(2), jnp.array([theta, 0.0]), com + jnp.array([0.55, 0.0]), jnp.zeros(2)]
        )
        return float(barriers.functions[0](0.0, z))

    h_facing = h_at(0.0)
    h_sideon = h_at(jnp.pi / 2)
    assert h_facing > 0.0 > h_sideon
    assert h_at(jnp.pi) == pytest.approx(h_facing, abs=1e-9)  # 180-degree symmetry


def test_hdi_wrapper_integrates_heading_and_commands_target_yaw(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import (
        com_agent_ellipse_hocbfs,
        embedded_heading_double_integrator,
        safe_locomotion_controller_hdi,
    )

    plant, g1mod = g1_plant
    x = g1mod.G1(plant.mj_model).x_stand(plant)

    class Drift:
        x0 = jnp.array([[30.0, 0.0, 0.0, 0.0]])  # far away: barriers inactive

        def step(self, t, robot_xy, states, dt):
            return states

    dyn = embedded_heading_double_integrator(plant.state_dim, plant.com_indices, n_agents=1)
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0, 2.0]),
        dynamics_func=dyn,
        barriers=com_agent_ellipse_hocbfs(plant, 1, (0.16, 0.28), ped_radius=0.30),
    )
    seen = {}

    def fake_loco(t, xx, cmd, key, d):
        seen["cmd"] = cmd
        return jnp.zeros(plant.nu), d

    ctrl = safe_locomotion_controller_hdi(cbf_qp, fake_loco, plant, 0.02, agents=Drift())
    d = ControllerData()
    for k in range(5):
        _, d = ctrl(k * 0.02, x, jnp.array([0.0, 0.4]), jax.random.PRNGKey(0), d)
    assert seen["cmd"].shape == (3,)
    th = float(d.sub_data["theta_cmd"])
    assert 0.0 < th <= jnp.pi / 2 + 1e-6  # turning toward +y (the travel direction)
    assert float(d.sub_data["v_safe"][1]) > 0.0  # accelerating along the command
    assert d.sub_data["a_safe"].shape == (3,)
    # 3-entry command overrides the heading target
    _, d2 = ctrl(0.1, x, jnp.array([0.0, 0.4, -1.0]), jax.random.PRNGKey(0), d)
    a_nom = jnp.asarray(d2.sub_data["a_nom"])
    assert a_nom[2] < 0.0  # steering toward the override, not the velocity direction


def _cost_traj(P, V, th=None, peds=((0.0, 0.5), (0.0, -0.5))):
    """(dim, H) states in the compact MPPI layout: [p v (th om)? | (p_i v_i) x N]."""
    H = P.shape[0]
    rows = [P.T, V.T]
    if th is not None:
        rows += [th[None, :], jnp.zeros((1, H))]
    for p in peds:
        rows.append(jnp.broadcast_to(jnp.array([p[0], p[1], 0.0, 0.0])[:, None], (4, H)))
    return jnp.concatenate(rows, axis=0)


def test_ellipse_trajectory_cost_ranks_the_homotopies():
    from cbfkit.systems.mujoco.reduced_order import ellipse_trajectory_cost

    H, dt = 21, 0.2
    goal = jnp.array([3.0, 0.0])
    xs = jnp.linspace(-1.5, 1.5, H)
    P = jnp.stack([xs, jnp.zeros(H)], axis=1)
    V = jnp.broadcast_to(jnp.array([0.3, 0.0]), (H, 2))
    U = jnp.zeros((3, H))
    kw = dict(heading=True, v_max=0.3)
    cost = ellipse_trajectory_cost(2, goal, (0.16, 0.28), 0.30, dt, lane_halfwidth=0.9, **kw)
    forward = float(cost(0.0, _cost_traj(P, V, th=jnp.zeros(H)), U))
    sideways = float(
        cost(0.0, _cost_traj(P, V, th=jnp.where(jnp.abs(xs) < 0.6, jnp.pi / 2, 0.0)), U)
    )
    # Turning sideways through the gap beats pushing through facing forward: the clearance
    # violation (the rotation signal) dominates the mild align/spin legibility terms.
    assert sideways < forward
    # A detour outside the lane loses to the sideways squeeze even without the lane term
    # (goal geometry alone), and the lane term prices it further up.
    P_det = jnp.stack([xs, jnp.full(H, 1.2)], axis=1)
    cost_nolane = ellipse_trajectory_cost(2, goal, (0.16, 0.28), 0.30, dt, **kw)
    det_nolane = float(cost_nolane(0.0, _cost_traj(P_det, V, th=jnp.zeros(H)), U))
    side_nolane = float(
        cost_nolane(0.0, _cost_traj(P, V, th=jnp.where(jnp.abs(xs) < 0.6, jnp.pi / 2, 0.0)), U)
    )
    assert side_nolane < det_nolane
    det_lane = float(cost(0.0, _cost_traj(P_det, V, th=jnp.zeros(H)), U))
    assert det_lane > det_nolane


def test_ellipse_cost_disc_variant_prices_the_tight_gap():
    from cbfkit.systems.mujoco.reduced_order import ellipse_trajectory_cost

    H, dt = 21, 0.2
    xs = jnp.linspace(-1.5, 1.5, H)
    P = jnp.stack([xs, jnp.zeros(H)], axis=1)
    V = jnp.broadcast_to(jnp.array([0.3, 0.0]), (H, 2))
    U = jnp.zeros((2, H))
    cost = ellipse_trajectory_cost(
        2, jnp.array([3.0, 0.0]), (0.35, 0.35), 0.30, dt, heading=False, v_max=0.3
    )
    tight = float(cost(0.0, _cost_traj(P, V), U))  # 1.0 m gap: dist 0.5 < 0.65 keep-out
    wide = float(cost(0.0, _cost_traj(P, V, peds=((0.0, 1.0), (0.0, -1.0))), U))
    assert jnp.isfinite(tight) and tight > wide  # the clearance term prices the tight gap


def test_mppi_local_planner_generalises_to_heading_dims():
    from cbfkit.systems.mujoco.reduced_order import mppi_local_planner
    from cbfkit.utils.user_types import PlannerData

    H = 6

    def fake_mppi(err):
        def mppi(t, xa, u, key, data):
            return jnp.array([0.1, 0.2, 0.3]), PlannerData(
                u_traj=jnp.ones((H, 3)), x_traj=jnp.ones((10, H + 1)), error=jnp.asarray(err)
            )

        return mppi

    planner = mppi_local_planner(
        fake_mppi(False), 1, horizon=H, replan_every=5, control_dim=3, state_head=6
    )
    xa = jnp.zeros(10).at[2].set(0.1)  # v = (0.1, 0)
    a0, v0, s0 = planner(0.0, xa, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), {})
    assert a0.shape == (3,) and jnp.allclose(a0, jnp.array([0.1, 0.2, 0.3]))
    assert s0["_mppi_u_traj"].shape == (H, 3) and s0["mppi_x_traj"].shape == (10, H + 1)
    assert v0.shape == (2,)
    # solver failure -> the P-law fallback, padded with a zero alpha channel
    p_fail = mppi_local_planner(
        fake_mppi(True), 1, horizon=H, replan_every=5, control_dim=3, state_head=6
    )
    a, _, _ = p_fail(0.0, xa, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), {})
    assert jnp.allclose(a, jnp.array([2.0 * (0.5 - 0.1), 0.0, 0.0]))


def test_hdi_wrapper_local_planner_hook_overrides_a_nom_and_carries_state(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import (
        com_agent_ellipse_hocbfs,
        embedded_heading_double_integrator,
        safe_locomotion_controller_hdi,
    )

    plant, g1mod = g1_plant
    x = g1mod.G1(plant.mj_model).x_stand(plant)
    ci = plant.com_indices

    class Drift:
        x0 = jnp.array([[30.0, 0.0, 0.0, 0.0]])  # far away: barriers inactive

        def step(self, t, robot_xy, states, dt):
            return states

    seen = {}

    def local_planner(t, xa_c, v_nom, key, sub):
        seen["xa_c"] = xa_c
        n = sub.get("_lp_calls")
        n = jnp.zeros((), dtype=jnp.int32) if n is None else n
        return (
            jnp.array([0.25, -0.25, 0.5]),
            jnp.array([0.1, 0.0]),
            {"_lp_calls": n + 1, "lp_flag": 1.0},
        )

    dyn = embedded_heading_double_integrator(plant.state_dim, ci, n_agents=1)
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0, 2.0]),
        dynamics_func=dyn,
        barriers=com_agent_ellipse_hocbfs(plant, 1, (0.16, 0.28), ped_radius=0.30),
    )

    def fake_loco(t, xx, cmd, key, d):
        return jnp.zeros(plant.nu), d

    ctrl = safe_locomotion_controller_hdi(
        cbf_qp, fake_loco, plant, 0.02, agents=Drift(), local_planner=local_planner
    )
    _, d1 = ctrl(0.0, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert seen["xa_c"].shape == (10,)  # [com | v | th om | one agent]
    assert jnp.allclose(seen["xa_c"][:2], x[ci[0] : ci[0] + 2])
    assert jnp.allclose(d1.sub_data["a_safe"], jnp.array([0.25, -0.25, 0.5]), atol=1e-3)
    assert jnp.allclose(d1.sub_data["v_nom"], jnp.array([0.1, 0.0]))
    assert int(d1.sub_data["_lp_calls"]) == 1 and float(d1.sub_data["lp_flag"]) == 1.0
    _, d2 = ctrl(0.02, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), d1)
    assert int(d2.sub_data["_lp_calls"]) == 2  # carried


def test_heading_social_cost_slices_the_layout_and_rewards_the_slim_profile():
    from cbfkit.controllers.mppi.social_costs import SocialCostWeights
    from cbfkit.systems.mujoco.reduced_order import heading_social_trajectory_cost

    H, dt, n = 15, 0.2, 1
    goal = jnp.array([5.0, 0.0])
    cost = heading_social_trajectory_cost(
        n, goal, SocialCostWeights(), dt, (0.16, 0.28), 0.30, robot_radius=0.35
    )
    xs = jnp.linspace(-0.7, 0.7, H)
    P = jnp.stack([xs, jnp.zeros(H)], axis=1)
    V = jnp.broadcast_to(jnp.array([0.5, 0.0]), (H, 2))
    ped = jnp.array([0.0, 0.35, 0.0, 0.0])  # standing 0.35 m to the side of the path

    def traj(th):
        rows = [P.T, V.T, th[None, :], jnp.zeros((1, H))]
        rows.append(jnp.broadcast_to(ped[:, None], (4, H)))
        return jnp.concatenate(rows, axis=0)

    U = jnp.zeros((3, H))
    c_fwd = float(cost(0.0, traj(jnp.zeros(H)), U))  # wide lateral axis toward the pedestrian
    c_side = float(cost(0.0, traj(jnp.full(H, jnp.pi / 2)), U))  # narrow axis toward her
    assert jnp.isfinite(c_fwd) and jnp.isfinite(c_side)
    # Passing 0.35 m from a person: facing forward points the 0.58 m lateral semi-axis at
    # her (deep ellipse violation); turned side-on the 0.46 m longitudinal axis does
    # (shallow). The ellipse clearance term must dominate the mild align/spin terms.
    assert c_side < c_fwd


@pytest.mark.parametrize("smoothing", [-0.1, 1.0, float("nan")])
def test_mppi_plan_smoothing_rejects_invalid_fraction(smoothing):
    from cbfkit.systems.mujoco.reduced_order import mppi_local_planner

    with pytest.raises(ValueError, match="plan_smoothing"):
        mppi_local_planner(None, 0, horizon=2, replan_every=2, plan_smoothing=smoothing)


def test_mppi_plan_smoothing_holds_nominal_and_recovers_from_failed_replan():
    from cbfkit.systems.mujoco.reduced_order import mppi_local_planner
    from cbfkit.utils.user_types import PlannerData

    def mppi(t, xa, u, key, data):
        value = jnp.where(t == 4, jnp.nan, t + 1)
        return jnp.full(3, value), PlannerData(
            u_traj=jnp.full((2, 3), value),
            x_traj=jnp.full((6, 3), value),
            error=t == 4,
        )

    planner = jax.jit(
        mppi_local_planner(
            mppi,
            0,
            horizon=2,
            replan_every=2,
            control_dim=3,
            state_head=6,
            plan_smoothing=0.75,
        )
    )
    carry = {}
    # First solve is unsmoothed. Replan at t=2 blends 75% of 1 with 25% of 3.
    # Failure at t=4 uses the finite P-law through the hold, then recovers at t=6.
    for t, expected in enumerate([1.0, 1.0, 1.5, 1.5, None, None, None]):
        a, _, carry = planner(
            jnp.asarray(t), jnp.zeros(6), jnp.zeros(2), jax.random.PRNGKey(0), carry
        )
        if t < 4:
            np.testing.assert_allclose(a, expected)
            np.testing.assert_allclose(carry["mppi_x_traj"], 1 if t < 2 else 3)
        elif t < 6:
            np.testing.assert_allclose(a, 0)
            assert carry["mppi_error"]
            np.testing.assert_allclose(carry["_mppi_u_traj"], 3)
        else:
            np.testing.assert_allclose(a, 0.25 * 7)
            assert not carry["mppi_error"]
        assert np.isfinite(carry["_mppi_a"]).all()
