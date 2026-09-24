"""Unit tests for the KUKA iiwa14 cell (``cbfkit.systems.mujoco.kuka_cell``).

Geometry, the worker script, the plant's control map, the pairwise barriers, the
operational-space nominal and the safe controller are exercised on the CPU with eager
JAX; the QP is called a handful of times only. The module needs the iiwa14 asset cache
and is skipped when it is absent (downloads are forbidden here).
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cbfkit.systems.mujoco import kuka_cell as kc  # noqa: E402
from cbfkit.utils.user_types import ControllerData  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "mujoco" / "kuka_cell.py"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CBFKIT_ASSETS_OFFLINE", "1")
        try:
            kc.load_cell(offline=True)
        except RuntimeError as exc:
            pytest.skip(f"iiwa14 assets unavailable: {exc}")
        yield


@pytest.fixture(scope="module")
def plant():
    return kc.make_plant(offline=True)


# --------------------------------------------------------------------------- Task 3
def test_worker_state_is_continuous_and_pauses():
    t = np.linspace(0, 12, 1201)
    s = np.stack([np.asarray(kc.worker_state(ti)) for ti in t])
    assert np.allclose(s[:, 0], kc.WORKER_X)
    assert np.all(np.abs(np.diff(s[:, 1])) <= kc.WORKER_SPEED * 0.01 + 1e-9)  # no jumps
    at_pause = np.isclose(s[:, 1], kc.WORKER_Y_PAUSE, atol=1e-6)
    assert at_pause.sum() * 0.01 >= kc.WORKER_PAUSE_S - 0.02
    assert np.allclose(s[0, 2:], 0.0)  # not yet entered
    assert np.isclose(s[-1, 1], kc.WORKER_Y0) and np.allclose(s[-1, 2:], 0.0)  # back out


def test_sphere_capsule_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c, a, b = rng.normal(size=(3, 3))
        r, rc = 0.05, 0.1
        ts = np.linspace(0, 1, 2001)
        brute = min(np.linalg.norm(c - (a + t * (b - a))) for t in ts) - r - rc
        assert abs(float(kc.sphere_capsule_distance(c, r, a, b, rc)) - brute) < 1e-4


def test_worker_capsules_rotate_with_yaw():
    a, b, r = kc.worker_capsules(jnp.array([1.0, 2.0]), jnp.pi / 2)
    assert a.shape == (5, 3) and b.shape == (5, 3) and r.shape == (5,)
    assert np.allclose(np.asarray(a)[1, :2], [1.0, 2.0])  # torso axis over the base point
    # facing +y: the left arm (body +y) ends up at world -x
    assert np.asarray(a)[3, 0] < 1.0 - 0.2


def test_load_cell_layout():
    m = kc.load_cell(offline=True)
    assert (m.nq, m.nv, m.nu, m.nmocap) == (7, 7, 7, 1)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool") >= 0
    assert abs(m.opt.timestep - kc.TIMESTEP) < 1e-12


# --------------------------------------------------------------------------- Task 4
def test_plant_layout_and_ctrl_map_moves_worker(plant):
    assert plant.nu == 14 and plant.state_dim == 17 and abs(plant.dt - 0.02) < 1e-12
    d = plant.make_data()
    pose = jnp.array([0.85, -1.0, 0.0])
    u = jnp.concatenate([jnp.zeros(7), pose, kc.worker_quat(jnp.pi / 2)])
    d2 = plant.step(d, u)
    assert np.allclose(np.asarray(d2.mocap_pos[0]), np.asarray(pose))
    assert np.allclose(np.asarray(d2.qpos), 0.0, atol=2e-3)  # servo holds the home pose
    # a held velocity command is tracked: after 0.3 s joint 1 moves at about qdot
    u2 = u.at[0].set(1.0)
    d3 = d2
    for _ in range(15):
        d3 = plant.step(d3, u2)
    assert 0.8 < float(d3.qvel[0]) < 1.2
    assert float(d3.ctrl[0]) - float(d3.qpos[0]) <= kc.CTRL_WINDUP + 1e-9
    # from_state seeds the servo target at the joint angles: no snap from a non-zero pose
    x = plant.to_state(plant.make_data().replace(qpos=jnp.asarray(kc.Q_HOME)))
    d4 = plant.step(plant.from_state(x), u.at[:7].set(0.0))
    assert np.allclose(np.asarray(d4.qpos), np.asarray(kc.Q_HOME), atol=5e-3)


def test_barriers_are_positive_far_and_negative_in_contact(plant):
    q = jnp.asarray(kc.Q_HOME)
    far = kc.barrier_values(plant, q, jnp.array([0.85, -2.4, 0.0, 0.0]))
    assert far.shape == (35,) and bool((far > 0).all())
    tool = np.asarray(kc.tool_pose(plant, q)[0])
    inside = kc.barrier_values(plant, q, jnp.array([tool[0], tool[1], 0.0, 0.0]))
    assert bool((inside < 0).any())
    assert len(kc.H_NAMES) == 35 and kc.H_NAMES[-1] == "tool/r_arm"


def test_certificate_collection_matches_barrier_values(plant):
    coll = kc.arm_worker_barriers(plant)
    assert len(coll.functions) == 35
    x = plant.to_state(plant.make_data().replace(qpos=jnp.asarray(kc.Q_HOME)))
    w = jnp.array([0.85, -0.3, 0.0, 1.1])
    xa = jnp.concatenate([x, w])
    h = np.array([float(f(0.0, xa)) for f in coll.functions])
    assert np.allclose(h, np.asarray(kc.barrier_values(plant, x[:7], w)), atol=1e-6)
    # the worker's motion enters through the embedded dynamics, not through dh/dt
    f, g = kc.embedded_arm_worker(plant.state_dim)(xa)
    assert f.shape == (21,) and g.shape == (21, 7)
    assert np.allclose(np.asarray(f[17:19]), [0.0, 1.1]) and np.allclose(np.asarray(g[:7]), np.eye(7))


# --------------------------------------------------------------------------- Task 5
def test_reference_speed_bounded_and_periodic():
    ts = np.arange(0, kc.CYCLE_S, 0.01)
    ref = jax.jit(kc.pick_place_reference)
    p = np.stack([np.asarray(ref(t)[0]) for t in ts])
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / 0.01
    assert speed.max() <= kc.V_TOOL + 0.05
    assert speed.max() > 0.9 * kc.V_TOOL  # the transfer actually swings at strike speed
    assert np.allclose(p[0], np.asarray(ref(kc.CYCLE_S)[0]), atol=1e-6)
    assert np.all(np.abs(np.diff(p, axis=0)).max(axis=1) < 0.03)  # continuous


def test_home_pose_holds_the_tool_over_the_bin(plant):
    p, a = kc.tool_pose(plant, jnp.asarray(kc.Q_HOME))
    assert np.allclose(np.asarray(p), [kc.BIN_XY[0], kc.BIN_XY[1], kc.Z_ABOVE_BIN], atol=0.02)
    assert float(a[2]) < -0.98  # pointing down


def test_osc_drives_tool_toward_reference(plant):
    nom = kc.osc_nominal(plant)
    q = jnp.asarray(kc.Q_HOME) + 0.15
    x = plant.to_state(plant.make_data().replace(qpos=q))
    p0, _ = kc.tool_pose(plant, q)
    pd, _ = kc.pick_place_reference(0.0)
    qd, _ = nom(0.0, x, None, None)
    assert qd.shape == (7,) and bool((jnp.abs(qd) <= jnp.asarray(kc.QD_LIMITS) + 1e-6).all())
    p1, _ = kc.tool_pose(plant, q + 0.05 * qd)
    assert float(jnp.linalg.norm(p1 - pd)) < float(jnp.linalg.norm(p0 - pd))


# --------------------------------------------------------------------------- Task 6
def test_safe_controller_passes_through_when_unfiltered():
    plant, x0, ctrl, nom = kc.build(unfiltered=True, offline=True)
    qd, _ = nom(0.0, x0, None, None)
    u, d = ctrl(0.0, x0, qd, jax.random.PRNGKey(0), ControllerData())
    assert u.shape == (14,) and not bool(d.error)
    lim = np.array(kc.QD_LIMITS)
    qd_nom = np.asarray(d.sub_data["qdot_nom"])
    assert np.allclose(qd_nom, np.asarray(qd), atol=1e-9)  # phase 0 == wall clock 0
    assert np.allclose(np.asarray(u[:7]), np.clip(qd_nom, -lim, lim), atol=1e-3)
    assert d.sub_data["h"].shape == (35,) and d.sub_data["worker"].shape == (4,)
    # the cycle clock advances at full rate while the tool tracks its reference
    assert abs(float(d.sub_data["_phase"]) - plant.dt) < 1e-9


def test_phase_holds_when_the_tool_is_far_from_its_reference(plant):
    q = jnp.asarray(kc.Q_HOME)
    assert float(kc.phase_rate(plant, q, 0.0)) == 1.0  # at bin_above, on the reference
    far_phase = kc._SEG_START[1] + 0.1  # reference is over the conveyor, a metre away
    assert float(kc.phase_rate(plant, q, far_phase)) == 0.0


def test_joint_limit_barriers_sign(plant):
    coll = kc.joint_limit_barriers(plant)
    assert len(coll.functions) == 14
    x = jnp.concatenate([plant.to_state(plant.make_data()), jnp.zeros(4)])
    h = np.array([float(f(0.0, x)) for f in coll.functions])
    assert (h > 0).all()  # q = 0 is inside every range
    x_hi = x.at[0].set(float(plant.q_max[0]))
    assert float(coll.functions[1](0.0, x_hi)) < 0  # joint 1 at its upper stop


def test_safe_controller_blocks_motion_toward_worker():
    plant, x0, ctrl, nom = kc.build(offline=True)
    # The worker is paused inside the sweep at this time; pick the joint-1 direction that
    # decreases the closest barrier and command it at full speed.
    t = kc.WORKER_T_ENTER + (kc.WORKER_Y_PAUSE - kc.WORKER_Y0) / kc.WORKER_SPEED + 0.5
    w = kc.worker_state(t)
    assert np.isclose(float(w[1]), kc.WORKER_Y_PAUSE) and np.allclose(np.asarray(w[2:]), 0.0)
    q = x0[:7]
    dh = jax.grad(lambda qq: kc.barrier_values(plant, qq, w).min())(q)
    sign = -np.sign(float(dh[0]))
    qd_nom = jnp.zeros(7).at[0].set(sign * kc.QD_LIMITS[0])
    u, d = ctrl(t, x0, qd_nom, jax.random.PRNGKey(0), ControllerData())
    assert not bool(d.error)
    h_min = float(np.asarray(d.sub_data["h"]).min())
    dh_dt_nom = float(dh[0] * qd_nom[0])
    # Either the constraint is inactive at this distance, or the QP cut the command.
    assert dh_dt_nom >= -kc.CLASS_K * h_min - 1e-6 or abs(float(u[0])) < kc.QD_LIMITS[0] - 1e-3


# --------------------------------------------------------------------------- Task 7
@pytest.mark.g1_mjx  # not the G1, but the same "compiles an MJX step + QP" cost class
def test_simulate_writes_showcase_npz(tmp_path):
    env = dict(
        os.environ,
        CBFKIT_TEST_MODE="1",
        JAX_PLATFORM_NAME="cpu",
        CBFKIT_ASSETS_OFFLINE="1",
        MPLBACKEND="Agg",
    )
    for flag in ([], ["--unfiltered"]):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "simulate", "--out", str(tmp_path), *flag],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
    with np.load(tmp_path / "kuka_cell.npz") as a:
        assert a["states"].shape == (5, 17) and a["h"].shape == (5, 35)
        assert str(a["plant_kind"]) == "iiwa14" and a["contact"].dtype == bool
        assert a["worker"].shape == (5, 4) and a["tool"].shape == (5, 3)
        assert a["com"].shape == (5, 2) and not bool(a["unfiltered"])
    with np.load(tmp_path / "kuka_cell_unfiltered.npz") as b:
        assert float(b["qp_dev_max"]) < 1e-3 and bool(b["unfiltered"])
        assert "bfs" not in b.files
