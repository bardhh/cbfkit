"""Unitree's pretrained G1 walking policy: torch-free loading, JAX evaluation, PD plant, walking."""

import urllib.error

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cbfkit.simulation.simulator as sim
from cbfkit.utils.user_types import ControllerData


@pytest.fixture(scope="module")
def up():
    mod = pytest.importorskip("cbfkit.systems.mujoco.unitree_policy")
    try:
        mod.unitree_rl_gym_dir()
    except (RuntimeError, urllib.error.URLError) as exc:
        pytest.skip(f"unitree_rl_gym assets unavailable: {exc}")
    return mod


def test_torchscript_loader_recovers_lstm_and_actor(up):
    params = up.load_g1_policy_params()
    assert params.w_ih.shape == (256, 47) and params.w_hh.shape == (256, 64)
    assert params.b_ih.shape == (256,) and params.b_hh.shape == (256,)
    assert params.w1.shape == (32, 64) and params.w2.shape == (12, 32)
    assert all(bool(jnp.all(jnp.isfinite(p))) for p in params)


def test_plant_dims_and_pd_map(up):
    plant = up.make_g1_12dof_plant()
    assert (plant.nq, plant.nv, plant.nu) == (19, 18, 12)
    assert plant.dt == pytest.approx(0.02)  # 2 ms x 10 substeps
    d = plant.make_data()
    q_target = jnp.asarray(up.G1_12DOF_CONFIG["default_angles"])
    d1 = plant.step(d, q_target)
    # PD pulls the joints toward the target within a control step.
    err0 = float(jnp.linalg.norm(d.qpos[7:19] - q_target))
    err1 = float(jnp.linalg.norm(d1.qpos[7:19] - q_target))
    assert err1 < err0


def test_policy_step_shapes_and_state_carry(up):
    plant = up.make_g1_12dof_plant()
    pol = up.UnitreeG1WalkPolicy()
    x = up.x0_standing(plant)
    s0 = pol.init_state()
    obs = pol.observation(x, jnp.array([0.5, 0.0, 0.0]), 0.0, s0.last_action)
    assert obs.shape == (47,)
    u, s1 = pol.step(x, jnp.array([0.5, 0.0, 0.0]), 0.0, s0)
    assert u.shape == (12,) and s1.h.shape == (64,)
    assert not jnp.allclose(s1.h, s0.h)  # LSTM state advanced
    # as_controller carries state in sub_data["_policy"] and is memoised.
    ctrl = pol.as_controller()
    assert ctrl is pol.as_controller()
    u2, d = ctrl(0.0, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert "_policy" in d.sub_data and jnp.allclose(u2, u)


@pytest.mark.slow
def test_policy_walks_forward_in_mjx(up):
    plant = up.make_g1_12dof_plant()
    ctrl = up.UnitreeG1WalkPolicy().as_controller()

    def nominal(t, x, key, ref):
        return jnp.array([0.5, 0.0]), ControllerData()

    res = sim.execute(
        x0=up.x0_standing(plant),
        dt=plant.dt,
        num_steps=150,
        plant=plant,
        nominal_controller=nominal,
        controller=ctrl,
        use_jit=True,
        verbose=False,
    )
    S = np.asarray(res["states"])
    vx = S[50:, 19]  # after 1 s of transient
    assert S[:, 2].min() > 0.6  # never falls (pelvis stays up)
    assert 0.35 < vx.mean() < 0.65  # tracks 0.5 m/s
    assert S[-1, 0] - S[0, 0] > 1.0  # actually travelled


@pytest.mark.slow
def test_navigate_example_certificate_holds(up):
    """Milestone 4/4b acceptance: robust CBF on the CoM + policy gait keeps h >= 0 and stays upright."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "mujoco" / "g1_navigate.py"
    spec = importlib.util.spec_from_file_location("g1_navigate", path)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    sim_plant, _loco, x0, _pb, (controller, nominal) = ex.build("policy", robust_bound=0.25)
    res = sim.execute(
        x0=x0,
        dt=sim_plant.dt,
        num_steps=300,
        plant=sim_plant,
        nominal_controller=nominal,
        controller=controller,
        use_jit=True,
        verbose=False,
    )  # 6 s: enough to reach the obstacle and start skirting it
    S = np.asarray(res["states"])
    ci = sim_plant.com_indices
    com = S[:, ci[0] : ci[0] + 2]
    r = ex.OBSTACLE_RADIUS + ex.ROBOT_RADIUS
    h = (
        ((com[:, 0] - float(ex.OBSTACLE[0])) / r) ** 2
        + ((com[:, 1] - float(ex.OBSTACLE[1])) / r) ** 2
        - 1
    )
    assert h.min() >= 0.0  # certificate holds throughout
    assert S[:, 2].min() > 0.6  # never falls
    assert com[-1, 0] > 1.0  # actually walked toward the goal
    v_safe = np.asarray(res.controller_data["sub_data_v_safe"])
    v_nom = np.asarray(res.controller_data["sub_data_v_nom"])
    assert np.any(np.linalg.norm(v_safe - v_nom, axis=1) > 1e-3)  # the CBF intervened


def test_command_frame_and_heading_follower(up):
    """World-frame command is rotated into the body frame; heading follower commands yaw toward it."""
    plant = up.make_g1_12dof_plant()
    pol = up.UnitreeG1WalkPolicy()
    x = up.x0_standing(plant)
    # Rotate the pelvis to yaw = +90 deg: quaternion (cos45, 0, 0, sin45).
    x = x.at[3:7].set(jnp.array([jnp.cos(jnp.pi / 4), 0.0, 0.0, jnp.sin(jnp.pi / 4)]))
    seen = {}
    orig_step = pol.step

    def spy(xx, cmd, t, state):
        seen["cmd"] = cmd
        return orig_step(xx, cmd, t, state)

    pol.step = spy
    ctrl = pol.as_controller(world_frame=True, heading_gain=2.0)
    ctrl(0.0, x, jnp.array([0.0, 0.5]), jax.random.PRNGKey(0), ControllerData())  # world +y
    cmd = np.asarray(seen["cmd"])
    assert (
        cmd[0] == pytest.approx(0.5, abs=1e-6) and abs(cmd[1]) < 1e-6
    )  # +y world == forward in body
    assert cmd[2] == pytest.approx(0.0, abs=1e-6)  # already facing the command: no yaw rate
    ctrl(
        0.0, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), ControllerData()
    )  # world +x = body right
    cmd = np.asarray(seen["cmd"])
    assert cmd[1] == pytest.approx(-0.5, abs=1e-6)  # strafe right in body frame
    assert cmd[2] < -0.5  # turn right (negative yaw rate) toward +x, clipped at max_yaw_rate
    body_ctrl = pol.as_controller(world_frame=False, heading_gain=0.0)
    body_ctrl(0.0, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert np.allclose(np.asarray(seen["cmd"]), [0.5, 0.0, 0.0])
    pol.step = orig_step


@pytest.mark.slow
def test_plaza_example_certificate_holds(up):
    """Plaza acceptance: pillar + reactive-pedestrian HOCBFs keep h >= 0 for every obstacle while
    the G1 completes the waypoint route upright (robust bound = the example's measured default),
    and the pedestrians demonstrably interact (come close, react)."""
    import importlib.util
    from pathlib import Path

    from cbfkit.utils.user_types import PlannerData

    path = Path(__file__).resolve().parents[2] / "examples" / "mujoco" / "g1_plaza.py"
    spec = importlib.util.spec_from_file_location("g1_plaza", path)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    plant, x0, _pb, planner, nominal, controller = ex.build(robust_bound=ex.DEFAULT_ROBUST_BOUND)
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=int(round(ex.DEFAULT_DURATION / plant.dt)),
        plant=plant,
        planner=planner,
        planner_data=PlannerData.from_constant(ex.WAYPOINTS[0]),
        nominal_controller=nominal,
        controller=controller,
        use_jit=True,
        verbose=False,
    )
    S = np.asarray(res["states"])
    ci = plant.com_indices
    com = S[:, ci[0] : ci[0] + 2]
    agents = np.asarray(res.controller_data["sub_data_agents"])
    H = ex.barrier_values(com, agents)
    assert H.min() >= 0.0, H.min(axis=0)  # every pillar and pedestrian keep-out respected
    assert S[:, 2].min() > 0.6  # never falls
    arrivals = ex.waypoint_arrivals(com)
    assert all(a is not None for a in arrivals), arrivals  # route completed
    n = arrivals[-1]
    ped_d = np.linalg.norm(com[:n, None, :] - agents[:n, :, :2], axis=2)
    assert ped_d.min(axis=0).max() < 1.6  # every pedestrian actually came close
    max_ped_acc = np.abs(np.gradient(agents[:n, :, 2:], plant.dt, axis=0)).max()
    assert max_ped_acc > 0.1  # and at least one of them reacted (accelerated) to the robot
    v_safe = np.asarray(res.controller_data["sub_data_v_safe"])
    v_nom = np.asarray(res.controller_data["sub_data_v_nom"])
    assert np.any(np.linalg.norm(v_safe - v_nom, axis=1) > 1e-3)  # the CBF intervened


@pytest.mark.slow
def test_scramble_example_crosses_without_contact(up):
    """Scramble acceptance (default: soft barriers, 40 pedestrians): the G1 crosses the
    intersection upright, never touches a pedestrian (CoM distance >= PED_RADIUS + 0.30, i.e.
    h >= -0.08 on the 0.65 m keep-out), and the crowd was genuinely around it."""
    import importlib.util
    from pathlib import Path

    from cbfkit.utils.user_types import PlannerData

    path = Path(__file__).resolve().parents[2] / "examples" / "mujoco" / "g1_scramble.py"
    spec = importlib.util.spec_from_file_location("g1_scramble", path)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    plant, x0, _pb, nominal, controller, _crowd = ex.build(  # full crowd even under TEST_MODE
        0, ex.DEFAULT_ROBUST_BOUND, ex.N_PED_FULL, ex.DEFAULT_RELAX
    )
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=int(round(ex.DEFAULT_DURATION / plant.dt)),
        plant=plant,
        planner_data=PlannerData.from_constant(ex.GOAL),
        nominal_controller=nominal,
        controller=controller,
        use_jit=True,
        verbose=False,
    )
    S = np.asarray(res["states"])
    ci = plant.com_indices
    com = S[:, ci[0] : ci[0] + 2]
    agents = np.asarray(res.controller_data["sub_data_agents"])
    d = np.linalg.norm(com[:, None, :] - agents[:, :, :2], axis=2)
    hit = np.flatnonzero(np.linalg.norm(com - np.asarray(ex.GOAL), axis=1) < ex.GOAL_RADIUS)
    assert hit.size, "crossing not completed"
    n = int(hit[0])
    assert d[:n].min() >= ex.PED_RADIUS + 0.30  # no contact (robot body radius ~0.30 at the CoM)
    assert S[:n, 2].min() > 0.6  # upright
    assert int(np.sum(d[:n].min(0) < 1.5)) >= 8  # it really went through the crowd
    assert not np.any(np.asarray(res.controller_data["error"])[:n])  # no QP failure


@pytest.mark.slow
def test_scramble_mppi_crosses_politely(up):
    """Social-MPPI acceptance on the full 40-pedestrian scramble (seed 0, MJX G1): crosses,
    never enters a keep-out disc, stays upright, and is measurably less intrusive than the
    goal-seeking baseline (measured there: intimate rate 4.2, front rate 2.0, CBF active 47%)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "mujoco" / "g1_scramble.py"
    spec = importlib.util.spec_from_file_location("g1_scramble", path)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    r = ex.run(90.0, 0, n_ped=ex.N_PED_FULL, relax=True, planner="mppi")
    m = r["metrics"]
    assert m["crossed"], "crossing not completed"
    assert m["h_min"] >= 0.0  # no pedestrian keep-out disc entered (measured +0.16)
    assert m["upright_min"] > 0.6
    assert m["intimate_rate"] < 3.0  # measured 1.7; goal baseline 4.2; human norm 3.5
    assert m["front_rate"] < 1.5  # measured 0.6; goal baseline 2.0; human norm 2.5
    assert m["cbf_active_frac"] < 0.35  # measured 19%; the plan is almost feasible as-is
    assert m["stopped_at_s"] is None and m["qp_nonconverged"] == 0


@pytest.mark.slow
def test_corridor_g1_sidesteps_through_a_gap_the_disc_refuses(up):
    """Anisotropic-footprint acceptance on the MJX G1 + AMO: at gap 1.25 m the ellipse
    CBF turns the robot sideways and it sidesteps through (measured: 102 s, theta 90 deg,
    h_min -0.16 = lateral tracking droop, upright 0.997); the disc CBF refuses the same
    gap (needs 1.30 m)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "mujoco" / "g1_corridor.py"
    spec = importlib.util.spec_from_file_location("g1_corridor", path)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    m, S, cd, plant = ex.run("ellipse", gap=ex.GAP_G1, g1=True, duration=120.0)
    assert m["crossed"], m
    assert m["theta_cmd_max_deg"] > 60.0  # turned sideways
    assert m["h_min"] > -0.30  # droop bounded by the measured tracking error
    assert m["min_centre_dist"] > ex.PED_RADIUS + 0.05  # no contact with the pedestrian disc
    assert m["upright_min"] > 0.9 and m["qp_errors"] == 0
    md, *_ = ex.run("disc", gap=ex.GAP_G1, g1=True, duration=60.0)
    # the disc certificate refuses the gap: parks before the gap line (x = 0), no crossing
    assert not md["crossed"] and md["x_max"] < 0.0
