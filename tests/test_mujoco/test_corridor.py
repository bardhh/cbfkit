"""The rotating-ellipse footprint certifies a squeeze the disc must refuse (g1_corridor)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[2] / "examples" / "mujoco" / "g1_corridor.py"
_spec = importlib.util.spec_from_file_location("g1_corridor", _path)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


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
    actual gap -- measured ~40% less heading travel and ~6x the clearance margin."""
    m_s, *_ = ex.run("ellipse", gap=1.0, duration=90.0, planner="suggest", offset=-0.35)
    m_p, *_ = ex.run("ellipse", gap=1.0, duration=90.0, planner="mppi", offset=-0.35)
    assert m_s["crossed"] and m_p["crossed"]
    assert m_p["theta_travel_deg"] < 0.8 * m_s["theta_travel_deg"]  # measured 107 vs 179
    assert m_p["h_min"] > m_s["h_min"] + 0.1  # measured +0.27 vs +0.04
