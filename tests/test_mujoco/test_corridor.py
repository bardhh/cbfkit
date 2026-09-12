"""The rotating-ellipse footprint certifies a squeeze the disc must refuse (g1_corridor)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[2] / "examples" / "mujoco" / "g1_corridor.py"
_spec = importlib.util.spec_from_file_location("g1_corridor", _path)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)
# These certificate checks run the measured configuration even under CBFKIT_TEST_MODE,
# whose smoke defaults cut the run to 5 steps and the MPPI planner to 64 samples.
ex.TEST_MODE = False
ex.MPPI_SAMPLES = 1024


def test_disc_refuses_the_gap_and_parks_at_the_boundary():
    m, S, cd, _ = ex.run("disc", gap=1.0, duration=40.0)
    assert not m["crossed"]
    assert m["x_max"] < -0.3  # never entered the gap
    assert m["min_centre_dist"] >= 0.65 - 5e-3  # rode the certified boundary
    assert m["qp_errors"] == 0


def test_ellipse_certifies_the_sidestep_through_the_same_gap():
    m, S, cd, plant = ex.run("ellipse", gap=1.0, duration=60.0)
    assert m["crossed"], m
    assert m["h_min"] >= 0.0  # hard-constraint certificate held on the commanded trajectory
    assert m["theta_cmd_max_deg"] > 60.0  # it actually turned sideways
    assert m["min_centre_dist"] < 0.60  # inside the disc's refusal radius: the squeeze happened
    assert m["qp_errors"] == 0
    # and it faces forward again after the squeeze
    th = np.asarray(cd["sub_data_theta_cmd"])
    n = int(m["time_s"] / plant.dt)
    assert abs(float(th[n - 1])) < 0.35


def test_wider_gap_needs_no_rotation():
    m, S, cd, _ = ex.run("ellipse", gap=1.6, duration=60.0)
    assert m["crossed"]
    assert m["h_min"] >= 0.0


def test_mppi_discovers_the_rotation_and_crosses_without_any_suggestion():
    """--planner mppi: no hand-coded heading anywhere -- the 6 s lookahead over the
    heading-augmented DI discovers that rotating pays, and the hard QP still holds h >= 0."""
    m, S, cd, _ = ex.run("ellipse", gap=1.0, duration=90.0, planner="mppi")
    assert m["crossed"], m
    assert m["h_min"] > 0.0
    assert m["theta_cmd_max_deg"] > 45.0  # discovered, not scripted
    assert m["qp_errors"] == 0 and m["mppi_errors"] == 0


def test_mppi_disc_still_refuses_under_the_identical_planner():
    m, S, cd, _ = ex.run("disc", gap=1.0, duration=60.0, planner="mppi")
    assert not m["crossed"]
    assert m["x_max"] < 0.0
    assert m["h_min"] > 0.0
    assert m["qp_errors"] == 0


def test_mppi_crosses_the_offset_gap_with_less_rotation_and_more_margin_than_the_ramp():
    """Gap centre shifted off the start-goal line: the fixed ramp turns the full 90 deg at
    its scripted spot and squeezes at h ~ 0.04; the lookahead plans a diagonal through the
    actual gap -- measured less rotation (68 vs 90 deg) at ~8x the clearance margin."""
    m_s, *_ = ex.run("ellipse", gap=1.0, duration=90.0, planner="suggest", offset=-0.35)
    m_p, *_ = ex.run("ellipse", gap=1.0, duration=90.0, planner="mppi", offset=-0.35)
    assert m_s["crossed"] and m_p["crossed"]
    assert m_p["theta_cmd_max_deg"] < m_s["theta_cmd_max_deg"]  # measured 68 vs 90
    assert m_p["h_min"] > m_s["h_min"] + 0.1  # measured +0.31 vs +0.04


@pytest.mark.parametrize("direction", [-1, 1])
def test_sidestep_commitment_holds_through_gap_then_releases_once(direction):
    import jax
    import jax.numpy as jnp

    def fake_planner(t, xa, v, key, sub):
        return jnp.array([0.1, 0.2, -direction]), jnp.array([0.1, 0.0]), {}

    planner = jax.jit(ex.committed_sidestep(fake_planner, 0.2, offset=0.1))
    key = jax.random.PRNGKey(0)
    state = jnp.array([-0.6, 0.1, 0.0, 0.0, direction * 0.8, 0.0])
    state = jnp.concatenate([state, ex.Static(1.5, offset=0.1).x0.reshape(-1)])
    a, _, carry = planner(0.0, state, jnp.zeros(2), key, {})
    assert carry["corridor_phase"] == 1
    assert direction * a[2] > 0  # complete the turn instead of following MPPI's early unwind
    # The pedestrian line and the old minimum-clearance location cannot release the turn.
    for x in [0.0, 0.25, 0.4]:
        state = state.at[0].set(x).at[4].set(direction * ex.SIDESTEP_HEADING)
        a, v, carry = planner(0.0, state, jnp.zeros(2), key, carry)
        assert carry["corridor_phase"] == 1
        assert abs(a[2]) < 1e-6
        assert v[0] > 0 and abs(v[1]) < 1e-6
    # Being past the release line is insufficient if too close to one pedestrian
    # for the full rotation sweep: keep the heading despite the forward progress.
    _, _, crowded = planner(0.0, state.at[0].set(0.5).at[1].set(0.8), jnp.zeros(2), key, carry)
    assert crowded["corridor_phase"] == 1
    state = state.at[0].set(0.5)
    a, _, carry = planner(0.0, state, jnp.zeros(2), key, carry)
    assert carry["corridor_phase"] == 2
    assert direction * a[2] < 0
    _, _, carry = planner(0.0, state.at[0].set(0.4), jnp.zeros(2), key, carry)
    assert carry["corridor_phase"] == 2  # no chattering on a backwards gait shuffle


def test_sidestep_commitment_leaves_approach_to_mppi():
    import jax
    import jax.numpy as jnp

    def fake_planner(t, xa, v, key, sub):
        return jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.0]), {}

    planner = ex.committed_sidestep(fake_planner, 0.2)
    for x, theta in [(-2.0, 0.9), (-0.5, 0.2), (0.1, 0.9)]:
        state = jnp.array([x, 0.0, 0.0, 0.0, theta, 0.0])
        state = jnp.concatenate([state, ex.Static(1.5).x0.reshape(-1)])
        a, _, carry = planner(0.0, state, jnp.zeros(2), jax.random.PRNGKey(0), {})
        np.testing.assert_allclose(a, [0.1, 0.2, 0.3])
        assert carry["corridor_phase"] == 0


def test_committed_sidestep_crosses_narrow_proxy_gap_with_hard_constraints():
    m, _, cd, _ = ex.run(
        "ellipse", gap=1.0, duration=60.0, planner="mppi", sidestep_commitment=True
    )
    assert m["crossed"] and m["h_min"] >= 0.0, m
    assert m["qp_errors"] == 0 and m["mppi_errors"] == 0
    np.testing.assert_array_equal(np.unique(cd["sub_data_corridor_phase"]), [0, 1, 2])
    phase = np.asarray(cd["sub_data_corridor_phase"])
    assert np.all(np.diff(phase) >= 0)
