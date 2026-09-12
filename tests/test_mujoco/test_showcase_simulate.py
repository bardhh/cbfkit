"""``g1_showcase.py simulate`` writes a self-contained npz the renderer can rely on.

Runs the navigate scenario twice in ``CBFKIT_TEST_MODE`` (5 steps) -- once filtered, once
with the empty certificate collection -- and checks the npz contract: the keys and shapes
of the logged arrays, the recomputed barrier against a hand computation at step 0, and the
pass-through invariant of the unfiltered run.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "mujoco" / "g1_showcase.py"

# Geometry of examples/mujoco/g1_navigate.py, written out rather than read from the npz so
# the check is independent of the metadata the driver stored.
OBSTACLE = np.array([2.0, 0.0])
KEEPOUT = 0.35 + 0.35  # OBSTACLE_RADIUS + ROBOT_RADIUS
A_MAX = 1.0

pytestmark = [pytest.mark.slow, pytest.mark.g1_mjx]


def _run(out_dir: Path, *args: str) -> Path:
    env = dict(os.environ, CBFKIT_TEST_MODE="1", MPLBACKEND="Agg", JAX_PLATFORM_NAME="cpu")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "simulate", "navigate", "--out", str(out_dir), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    suffix = "_unfiltered" if "--unfiltered" in args else ""
    path = out_dir / f"g1_navigate{suffix}.npz"
    assert path.exists(), proc.stdout[-2000:]
    return path


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("showcase")
    return {
        "cbf": np.load(_run(out), allow_pickle=False),
        "unfiltered": np.load(_run(out, "--unfiltered"), allow_pickle=False),
    }


def test_npz_carries_the_documented_keys_and_shapes(runs):
    z = runs["cbf"]
    t = int(z["states"].shape[0])
    assert t == 5  # CBFKIT_TEST_MODE horizon
    assert z["states"].shape == (t, 40)  # 12-DoF plant: qpos 19 + qvel 18 + com 3
    assert str(z["plant_kind"]) == "unitree12"
    assert (int(z["nq"]), int(z["nv"])) == (19, 18)
    assert tuple(z["com_indices"]) == (37, 38)
    assert z["com"].shape == (t, 2)
    assert int(z["pelvis_body"]) >= 0
    assert 0 <= int(z["n_live"]) <= t
    assert z["h"].shape == (t, 1)
    assert list(z["h_names"]) == ["obstacle"]
    assert str(z["example"]) == "navigate"
    assert not bool(z["unfiltered"])
    assert str(z["unfiltered_mode"]) == ""  # empty on a filtered run
    assert int(z["n_u"]) == 2
    assert not bool(z["relax"])
    assert float(z["dt"]) == pytest.approx(0.02)
    # controller log: the di wrapper's own keys plus the ControllerData fields
    for key, shape in {
        "v_nom": (t, 2),
        "v_safe": (t, 2),
        "a_nom": (t, 2),
        "a_safe": (t, 2),
        "bfs": (t, 1),
        "u": (t, 12),
        "u_nom": (t, 2),
        "sol": (t, 2),
    }.items():
        assert z[key].shape == shape, key
    for key in ("violated", "solver_status", "solver_iter", "error", "complete"):
        assert z[key].shape == (t,), key
    # scenario metadata
    assert z["obstacles"].shape == (1, 2)
    assert z["obstacle_radii"].shape == (1,)
    assert z["goal"].shape == (2,)
    assert z["waypoints"].shape == (0, 2)
    assert z["footprint_axes"].shape == (2,)
    assert float(z["ped_radius"]) == 0.0
    assert int(z["seed"]) == 0
    assert np.allclose(z["obstacles"][0], OBSTACLE)
    # no pedestrians and no MPPI planner in this scenario
    assert "agents" not in z.files
    assert "mppi_x_traj" not in z.files


def test_recomputed_h_matches_a_hand_computation(runs):
    z = runs["cbf"]
    com = z["states"][:, 37:39]
    assert np.allclose(com, z["com"])
    expected = ((com[0] - OBSTACLE) / KEEPOUT) ** 2
    assert float(z["h"][0, 0]) == pytest.approx(float(expected.sum() - 1.0), abs=1e-12)
    # h is the barrier, not the logged psi_1 of the rectified high-order cascade
    assert not np.allclose(z["h"][:, 0], z["bfs"][:, 0])


def test_unfiltered_run_is_a_pass_through(runs):
    z = runs["unfiltered"]
    assert bool(z["unfiltered"])
    assert str(z["unfiltered_mode"]) == "planner"
    assert "bfs" not in z.files  # the empty collection assembles no certificate rows
    assert "violated" not in z.files
    # With no certificates the QP is the projection of the nominal onto the control box, so
    # the deviation is solver noise (jaxopt OSQP, g1_navigate's default: ~1e-4).
    dev = np.abs(z["a_safe"] - np.clip(z["a_nom"], -A_MAX, A_MAX)).max()
    assert dev < 1e-3, dev
    assert float(z["qp_dev_max"]) == pytest.approx(dev, rel=1e-9)
    # ... and that is three orders of magnitude below the filtered run's intervention
    filtered = runs["cbf"]
    assert float(filtered["qp_dev_max"]) > 100 * dev
    # v_safe is the *integral* of the certified acceleration, so it never equals v_nom
    assert z["v_safe"].shape == z["v_nom"].shape
    assert z["h"].shape == (z["states"].shape[0], 1)
