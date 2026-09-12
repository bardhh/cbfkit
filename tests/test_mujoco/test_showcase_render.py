"""Tests for ``examples/mujoco/g1_showcase_render.py`` on synthetic npz files.

No simulation is involved: each test writes an npz with the exact key layout
``g1_showcase.py simulate`` produces (the G1 home pose repeated, a couple of fake
pedestrians, a fake MPPI plan and a fake ``h``), then renders it. That exercises the model
build, the overlay set of every example, the camera, the HUD and both encoders, which is
everything the render driver owns; the numbers in the frames are meaningless by design.

The whole module needs the G1 asset cache (the scene XML and its meshes) and is skipped
when it is absent. Downloads are forbidden here: ``CBFKIT_ASSETS_OFFLINE=1`` is set for the
module, so a cold cache skips rather than pulling ~50 meshes inside a test.
"""

import os
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cbfkit.systems.mujoco import showcase  # noqa: E402

pytestmark = pytest.mark.slow

DT = 0.02
N_STEPS = 30
N_PED = 2
HORIZON = 8
GAP = 8  # compose_panels' default gap between the rows


@pytest.fixture(scope="module", autouse=True)
def _offline():
    """Forbid asset downloads for the whole module (a cold cache must skip, not fetch)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CBFKIT_ASSETS_OFFLINE", "1")
        yield


@pytest.fixture(scope="module")
def g1_model():
    try:
        return showcase.render_model("unitree12", offline=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        pytest.skip(f"G1 render model unavailable (asset cache): {exc}")


@pytest.fixture(scope="module")
def render_module(g1_model):
    """Imported only once the model is known to build, so the skip fires first."""
    from examples.mujoco import g1_showcase_render

    return g1_showcase_render


def _has_ffmpeg() -> bool:
    try:
        showcase.ffmpeg_exe()
    except RuntimeError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="no ffmpeg available")


def _probe(path):
    """``ffmpeg -i <path>``'s stream table (printed on stderr; ffmpeg exits 1 with no output)."""
    import subprocess

    done = subprocess.run(
        [showcase.ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True
    )
    return done.stderr.decode(errors="replace")


# --------------------------------------------------------------------------- fixtures
def _base_payload(model, example: str, *, unfiltered: bool = False) -> dict:
    """A run with the G1 standing pose translated along a gentle S, plus fake controller logs.

    ``unitree12``'s scene has no keyframes (``nkey == 0``), so the standing pose is the
    model's ``qpos0`` -- which is exactly what ``unitree_policy.x0_standing`` uses.
    """
    nq, nv = int(model.nq), int(model.nv)
    t = np.linspace(0.0, 1.0, N_STEPS)
    com = np.stack([3.0 * t, 0.3 * np.sin(2.0 * np.pi * t)], axis=1)

    states = np.zeros((N_STEPS, nq + nv + 3))
    states[:, :nq] = np.asarray(model.qpos0, dtype=float)
    states[:, 0:2] = com  # free-joint translation follows the CoM
    states[:, nq + nv : nq + nv + 2] = com
    states[:, nq + nv + 2] = float(model.qpos0[2])

    agents = np.zeros((N_STEPS, N_PED, 4))
    for j in range(N_PED):
        agents[:, j, 0] = 1.5 + 0.5 * j
        agents[:, j, 1] = 0.8 * (-1.0) ** j + 0.2 * t
        agents[:, j, 3] = 0.2 * (-1.0) ** j
    h = np.linalg.norm(com[:, None, :] - agents[:, :, :2], axis=2) / 0.65 - 1.0

    plan = np.zeros((N_STEPS, 4 + 4 * N_PED, HORIZON + 1))
    for k in range(N_STEPS):
        plan[k, 0] = com[k, 0] + np.linspace(0.0, 1.0, HORIZON + 1)
        plan[k, 1] = com[k, 1]

    v_nom = np.tile(np.array([0.5, 0.0]), (N_STEPS, 1))
    v_safe = v_nom + np.stack([np.zeros(N_STEPS), 0.1 * np.sin(6.0 * t)], axis=1)
    solver_status = np.ones(N_STEPS, dtype=np.int64)
    solver_status[5] = 0  # exercises the "QP not converged" pill
    mppi_error = np.zeros(N_STEPS, dtype=bool)
    mppi_error[7] = True

    return dict(
        states=states,
        com=com,
        dt=np.float64(DT),
        plant_kind=np.array("unitree12"),
        nq=np.int64(nq),
        nv=np.int64(nv),
        com_indices=np.asarray([nq + nv, nq + nv + 1, nq + nv + 2], dtype=np.int64),
        pelvis_body=np.int64(1),
        n_live=np.int64(N_STEPS - 6),
        h=h,
        h_names=np.asarray([f"pedestrian {j + 1}" for j in range(N_PED)]),
        agents=agents,
        v_nom=v_nom,
        v_safe=v_safe,
        a_nom=np.zeros((N_STEPS, 2)),
        a_safe=np.stack([np.zeros(N_STEPS), 0.2 * np.cos(4.0 * t)], axis=1),
        mppi_x_traj=plan,
        mppi_error=mppi_error,
        solver_status=solver_status,
        solver_iter=np.full(N_STEPS, 12, dtype=np.int64),
        sol=np.zeros((N_STEPS, 4)),
        error=np.zeros(N_STEPS, dtype=bool),
        complete=np.zeros(N_STEPS, dtype=bool),
        example=np.array(example),
        unfiltered=np.bool_(unfiltered),
        seed=np.int64(0),
        n_u=np.int64(2),
        relax=np.bool_(True),
        goal=np.array([3.0, 0.0]),
        goal_radius=np.float64(0.4),
        ped_radius=np.float64(0.30),
        ped_keepout=np.float64(0.65),
        robot_radius=np.float64(0.35),
        footprint_axes=np.array([0.11, 0.22]),
        obstacles=np.zeros((0, 2)),
        obstacle_radii=np.zeros(0),
        keepout_radii=np.zeros(0),
        waypoints=np.zeros((0, 2)),
        barrier_shape=np.array("distance"),
    )


def _payload(model, example: str, *, unfiltered: bool = False) -> dict:
    """``_base_payload`` specialised to one example's metadata (as the simulate driver writes it)."""
    payload = _base_payload(model, example, unfiltered=unfiltered)
    if example == "navigate":
        payload.update(
            obstacles=np.array([[1.5, 0.0]]),
            obstacle_radii=np.array([0.35]),
            keepout_radii=np.array([0.70]),
            h=payload["h"][:, :1],
            h_names=np.asarray(["obstacle"]),
            barrier_shape=np.array("ellipsoid"),  # the quadratic form (d/r)^2 - 1
            relax=np.bool_(False),
        )
        del payload["agents"]
    elif example == "plaza":
        payload.update(
            obstacles=np.array([[1.0, 0.6], [2.2, -0.5]]),
            obstacle_radii=np.full(2, 0.30),
            keepout_radii=np.full(2, 0.65),
            waypoints=np.array([[1.5, 0.8], [3.0, 0.0]]),
            x_traj=np.tile(np.array([[1.5], [0.8]]), (N_STEPS, 1, 1)),
            h=np.concatenate([payload["h"], payload["h"]], axis=1),  # 2 pillars + 2 pedestrians
            h_names=np.asarray(["pillar 1", "pillar 2", "pedestrian 1", "pedestrian 2"]),
            relax=np.bool_(False),
        )
    elif example == "corridor":
        payload.update(
            obstacles=payload["agents"][0, :, :2],
            obstacle_radii=np.full(N_PED, 0.30),
            keepout_radii=np.full(N_PED, 0.65),
            footprint_axes_inflated=np.array([0.41, 0.52]),
            theta_cmd=0.3 * np.sin(np.linspace(0.0, 3.0, N_STEPS)),
            barrier_shape=np.array("ellipse"),
            n_u=np.int64(3),
            relax=np.bool_(False),
        )
    elif example == "scramble":
        payload.update(half=np.float64(6.0))
    if unfiltered:
        # An empty certificate collection assembles no rows: `bfs`/`violated` leave the log,
        # and a relaxed run loses its slack columns, so `sol` narrows to (T, n_u).
        payload["sol"] = np.zeros((N_STEPS, int(payload["n_u"])))
    else:
        payload["bfs"] = payload["h"].copy()
        payload["violated"] = np.zeros(N_STEPS, dtype=bool)
    return payload


def _write(tmp_path, model, example: str, *, unfiltered: bool = False, mode: str = "") -> str:
    name = f"g1_{example}{'_unfiltered' if unfiltered else ''}.npz"
    path = str(tmp_path / name)
    payload = _payload(model, example, unfiltered=unfiltered)
    if mode:  # the simulate driver's `unfiltered_mode`: what still drives the robot
        payload["unfiltered_mode"] = np.array(mode)
    np.savez_compressed(path, **payload)
    return path


def _size(path):
    from PIL import Image

    with Image.open(path) as img:
        return img.size  # (width, height)


# --------------------------------------------------------------------------- loading
def test_run_reads_the_npz_layout(render_module, g1_model, tmp_path):
    run = render_module.Run(_write(tmp_path, g1_model, "scramble"))
    assert run.example == "scramble"
    assert run.plant_kind == "unitree12"
    assert run.T == N_STEPS
    assert run.dt == pytest.approx(DT)
    assert run.h_min.shape == (N_STEPS,)
    assert run.get("agents").shape == (N_STEPS, N_PED, 4)
    assert run.get("not_logged_here") is None
    assert run.ped_keepout == pytest.approx(0.65)
    assert run.relax is True


def test_intervention_prefers_the_filtered_variable(render_module, g1_model, tmp_path):
    """``a`` is the certified variable for the DI wrappers, so it is what the bar reports."""
    run = render_module.Run(_write(tmp_path, g1_model, "scramble"))
    values, caption, scale = render_module._intervention(run)
    assert values.shape == (N_STEPS,)
    assert "m/s^2" in caption
    assert scale == pytest.approx(2.0)  # 2 * a_max, not the 0.5 m/s velocity scale
    expected = np.linalg.norm(run.get("a_safe") - run.get("a_nom"), axis=1)
    assert np.allclose(values, expected)


def test_unfiltered_relaxed_run_has_no_slack_columns(render_module, g1_model, tmp_path):
    """Removing the certificates removes the slack columns, so the pill must simply vanish."""
    filtered = render_module.Run(_write(tmp_path, g1_model, "scramble"))
    unfiltered = render_module.Run(_write(tmp_path, g1_model, "scramble", unfiltered=True))
    assert filtered.relax and unfiltered.relax
    assert unfiltered.get("sol").shape == (N_STEPS, unfiltered.n_u)
    assert unfiltered.get("bfs") is None and unfiltered.get("violated") is None
    iv, caption, fs = render_module._intervention(unfiltered)
    assert render_module._hud_kwargs(unfiltered, "scramble", 3, iv, caption, fs)["slack"] is None
    iv, caption, fs = render_module._intervention(filtered)
    assert render_module._hud_kwargs(filtered, "scramble", 3, iv, caption, fs)["slack"] == 0.0


@pytest.mark.parametrize(
    "example, label",
    [
        ("navigate", "Walking policy only"),
        ("plaza", "Walking policy only"),
        ("corridor", "MPPI planner, no certificate"),
        ("scramble", "MPPI planner, no certificate"),
    ],
)
def test_unfiltered_label_matches_what_the_comparison_run_actually_is(
    render_module, g1_model, tmp_path, example, label
):
    """corridor/scramble keep their MPPI planner without certificates; the others do not."""
    assert render_module.LABEL_UNFILTERED[example] == label
    run = render_module.Run(_write(tmp_path, g1_model, example, unfiltered=True))
    assert render_module._mode(run, example) == label.lower()


def test_unfiltered_mode_key_overrides_the_example_label(render_module, g1_model, tmp_path):
    """``unfiltered_mode`` says what is still driving the robot; it beats the example map."""
    nominal = render_module.Run(
        _write(tmp_path, g1_model, "corridor", unfiltered=True, mode="nominal")
    )
    assert render_module._unfiltered_label(nominal, "corridor") == "Goal-directed policy, no filter"
    planner = render_module.Run(
        _write(tmp_path, g1_model, "plaza", unfiltered=True, mode="planner")
    )
    assert render_module._unfiltered_label(planner, "plaza") == "MPPI planner, no certificate"
    # an npz from before the key exists falls back to the per-example map
    old = render_module.Run(_write(tmp_path, g1_model, "navigate", unfiltered=True))
    assert old.arrays.get("unfiltered_mode") is None
    assert render_module._unfiltered_label(old, "navigate") == "Walking policy only"
    # an unrecognised value is not silently turned into a wrong claim
    odd = render_module.Run(_write(tmp_path, g1_model, "scramble", unfiltered=True, mode="???"))
    assert render_module._unfiltered_label(odd, "scramble") == "MPPI planner, no certificate"


def _record_crowd(render_module, monkeypatch):
    """Capture who ``_crowd`` draws and rings, instead of measuring it through geom counts."""
    drawn, ringed = [], []

    def person(scn, p, v, r, body_scale=None, height=None):
        drawn.append((float(p[0]), body_scale, height))
        return 1

    monkeypatch.setattr(render_module, "_person", person)
    monkeypatch.setattr(render_module, "_keepout", lambda scn, p, r, h: ringed.append(float(p[0])))
    return drawn, ringed


def test_scramble_culls_the_far_crowd_and_rings_only_the_nearest(
    render_module, g1_model, tmp_path, monkeypatch
):
    """40 people at 0.8 radius screen the robot off; only the near ones earn a body and a ring."""
    run = render_module.Run(_write(tmp_path, g1_model, "scramble"))
    n = 12
    agents = np.zeros((N_STEPS, n, 4))
    agents[:, :, 0] = run.com[0, 0] + np.arange(1.0, n + 1.0)  # 1 m .. 12 m straight ahead
    agents[:, :, 1] = run.com[0, 1]
    run.arrays["agents"] = agents
    run.h_display = np.zeros((N_STEPS, n))
    drawn, ringed = _record_crowd(render_module, monkeypatch)
    scene = mujoco.MjvScene(g1_model, maxgeom=4096)
    render_module._overlay_scramble(scene, run, 0, render_module._prepare(run, "scramble"))
    assert [round(x) for x in sorted(d[0] for d in drawn)] == list(range(1, 10))  # 10-12 culled
    assert {d[1] for d in drawn} == {render_module.PED_BODY_SCALE_CROWD}
    assert {d[2] for d in drawn} == {render_module.PED_HEIGHT_CROWD}
    assert len(ringed) == render_module.N_RING_NEAR == 4
    assert sorted(round(x) for x in ringed) == [1, 2, 3, 4]


def test_plaza_keeps_the_wide_pedestrians_and_no_cull(
    render_module, g1_model, tmp_path, monkeypatch
):
    run = render_module.Run(_write(tmp_path, g1_model, "plaza"))
    agents = np.zeros((N_STEPS, 2, 4))
    agents[:, 0, 0] = run.com[0, 0] + 1.0
    agents[:, 1, 0] = run.com[0, 0] + 20.0  # far away, but plaza draws everyone
    run.arrays["agents"] = agents
    drawn, _ringed = _record_crowd(render_module, monkeypatch)
    scene = mujoco.MjvScene(g1_model, maxgeom=4096)
    render_module._overlay_plaza(scene, run, 0, render_module._prepare(run, "plaza"))
    assert len(drawn) == 2
    assert {d[1] for d in drawn} == {render_module.PED_BODY_SCALE} == {0.8}


def test_scramble_camera_looks_down_into_the_crowd(render_module):
    assert render_module.CAMERAS["scramble"] == (6.5, -40.0)
    assert render_module.CAMERAS["navigate"] == (3.2, -18.0)  # the others are unchanged


def test_quadratic_barrier_is_displayed_in_the_distance_form(render_module, g1_model, tmp_path):
    """navigate's h is (d/r)^2 - 1; the HUD shows the equivalent d/r - 1."""
    run = render_module.Run(_write(tmp_path, g1_model, "navigate"))
    assert run.barrier_shape == "ellipsoid"
    assert run.distance_form is True
    assert np.allclose(run.h_display, np.sqrt(run.h + 1.0) - 1.0)
    assert np.allclose(run.h_min, run.h_display.min(axis=1))
    # the reparametrisation is monotone and keeps the zero crossing, so the sign never moves
    assert np.array_equal(np.sign(run.h_display), np.sign(run.h))
    iv, caption, fs = render_module._intervention(run)
    assert render_module._hud_kwargs(run, "navigate", 5, iv, caption, fs)["h_label"] == (
        "h_min (distance form)"
    )


@pytest.mark.parametrize("example", ["plaza", "corridor", "scramble"])
def test_distance_shaped_barriers_are_left_alone(render_module, g1_model, tmp_path, example):
    run = render_module.Run(_write(tmp_path, g1_model, example))
    assert run.distance_form is False
    assert np.array_equal(run.h_display, run.h)
    iv, caption, fs = render_module._intervention(run)
    assert render_module._hud_kwargs(run, example, 5, iv, caption, fs)["h_label"] == "h_min"


# ------------------------------------------------------------------- side-by-side cameras
def test_each_panel_tracks_its_own_robot(render_module, g1_model, tmp_path):
    """A shared look-at leaves one panel on empty floor once the two runs diverge."""
    cbf = render_module.Run(_write(tmp_path, g1_model, "scramble"))
    other = render_module.Run(_write(tmp_path, g1_model, "scramble", unfiltered=True))
    other.com = cbf.com + np.array([8.0, 0.0])  # the unfiltered robot walked off to the side
    a = render_module._camera_path(cbf, "scramble", 0, N_STEPS, render_module.DRIFT_DEG_PER_S)
    b = render_module._camera_path(other, "scramble", 0, N_STEPS, render_module.DRIFT_DEG_PER_S)
    assert a.shape == b.shape == (N_STEPS, 4)
    assert np.allclose(b[:, 0] - a[:, 0], 8.0)  # each look-at follows its own CoM
    assert np.allclose(a[:, 3], b[:, 3])  # same azimuth schedule, so the shots stay comparable


def test_camera_path_is_independent_of_which_outputs_are_written(render_module, g1_model, tmp_path):
    """The MP4's move must not change because a GIF was also asked for."""
    run = render_module.Run(_write(tmp_path, g1_model, "navigate"))
    path = render_module._camera_path(run, "navigate", 0, N_STEPS, render_module.DRIFT_DEG_PER_S)
    # sampling at the MP4 stride gives the same rows whatever else is encoded
    assert np.allclose(path[::2], path[[k for k in range(0, N_STEPS, 2)]])
    assert len(path) == N_STEPS  # one row per logged step, not per encoded frame


def test_frozen_panel_reports_why_its_run_ended(render_module, g1_model, tmp_path):
    run = render_module.Run(_write(tmp_path, g1_model, "scramble"))
    panel_end = int(run.n_live)
    assert run.clip(panel_end) == panel_end
    caption = render_module._end_caption(run)
    assert caption.startswith(("goal reached", "run ended"))
    assert f"{panel_end * run.dt:.1f} s" in caption
    # the CoM at n_live is 2.4 m from the goal in this fixture, so it is not a goal hit
    assert caption.startswith("run ended")
    run.com = np.tile(run.goal, (run.T, 1))  # now it is
    assert render_module._end_caption(run).startswith("goal reached")


def test_compose_panels_draws_a_caption_under_the_label():
    panels = [np.full((200, 300, 3), 90, np.uint8), np.full((200, 300, 3), 90, np.uint8)]
    plain = showcase.compose_panels(panels, ["a", "b"], None)
    captioned = showcase.compose_panels(panels, ["a", "b"], None, captions=["run ended", None])
    assert captioned.shape == plain.shape
    assert not np.array_equal(plain, captioned)  # the left panel gained a second pill
    assert np.array_equal(plain[:, 300:], captioned[:, 300:])  # the right one did not
    with pytest.raises(ValueError):
        showcase.compose_panels(panels, ["a", "b"], None, captions=["only one"])


# --------------------------------------------------------------------------- window / ylim
def test_window_indices_slice_clip_and_fall_back(render_module):
    w = render_module._window_indices
    assert w(1000, 0.02, (2.0, 6.0), None) == (100, 300, True)
    assert w(1000, 0.02, None, None) == (0, 1000, False)
    assert w(1000, 0.02, (2.0, 60.0), None) == (100, 1000, True)  # clipped to the live part
    assert w(1000, 0.02, (2.0, 6.0), 1.0) == (100, 150, True)  # max_seconds counts from the start
    # a window past the end is dropped, and reported as not applied so the stills clamp to n_live
    assert w(50, 0.02, (20.0, 56.0), None) == (0, 50, False)
    assert w(50, 0.02, (0.4, 0.2), None) == (0, 50, False)  # empty window: likewise


def test_window_defaults_match_the_brief(render_module):
    assert render_module.WINDOWS == {
        "navigate": None,
        "plaza": None,
        "corridor": (20.0, 56.0),
        "scramble": (6.0, 42.0),
    }


def test_hud_ylim_is_pinned_by_the_rendered_stretch(render_module, g1_model, tmp_path):
    run = render_module.Run(_write(tmp_path, g1_model, "scramble"))
    run.h_min = np.concatenate([np.full(90, 7.0), np.full(10, 0.2)])  # navigate-like opening
    lo, hi = render_module._hud_ylim(run, 0, 100)
    assert lo == pytest.approx(-0.25)  # never above the zero line
    assert hi == pytest.approx(7.0)  # the 90th percentile, not the max
    run.h_min = np.linspace(0.8, -0.6, 100)
    lo, hi = render_module._hud_ylim(run, 0, 100)
    assert lo == pytest.approx(1.1 * -0.6)
    assert hi == pytest.approx(1.0)  # the floor, so a shallow clip still shows some headroom
    run.h_min = np.linspace(0.4, 0.1, 100)  # corridor: the whole trace must stay readable
    assert render_module._hud_ylim(run, 0, 100) == pytest.approx((-0.25, 1.0))


def test_render_honours_an_explicit_window(render_module, g1_model, tmp_path):
    out = tmp_path / "out"
    paths = render_module.render(
        "navigate",
        _write(tmp_path, g1_model, "navigate"),
        out_dir=str(out),
        stills=True,
        window=(0.1, 0.3),
    )
    assert len(paths) == 5
    assert all(os.path.getsize(p) > 0 for p in paths)
    with pytest.raises(ValueError, match="t0, t1"):
        render_module.render(
            "navigate",
            paths and _write(tmp_path, g1_model, "navigate"),
            out_dir=str(out),
            stills=True,
            window="whenever",
        )


def test_only_plaza_uses_x_traj_as_the_beacon(render_module, g1_model, tmp_path):
    """corridor/scramble also log ``x_traj``, but theirs is the constant goal."""
    import inspect

    for example in ("navigate", "corridor", "scramble"):
        source = inspect.getsource(render_module.OVERLAYS[example])
        assert "x_traj" not in source
    assert "x_traj" in inspect.getsource(render_module.OVERLAYS["plaza"])


def test_render_rejects_a_mismatched_example(render_module, g1_model, tmp_path):
    npz = _write(tmp_path, g1_model, "navigate")
    with pytest.raises(ValueError, match="simulated as"):
        render_module.render("plaza", npz, out_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="unfiltered-npz"):
        render_module.render("navigate", npz, out_dir=str(tmp_path / "out"), side_by_side=True)


# --------------------------------------------------------------------------- stills
@pytest.mark.parametrize("example", ["navigate", "plaza", "corridor", "scramble"])
def test_stills_cover_every_overlay_set(render_module, g1_model, tmp_path, example):
    out = tmp_path / "out"
    paths = render_module.render(
        example,
        _write(tmp_path, g1_model, example),
        out_dir=str(out),
        side_by_side=False,
        stills=True,
        max_seconds=0.3,
    )
    assert len(paths) == 5
    for fraction, path in zip((0, 25, 50, 75, 100), paths):
        assert path.endswith(f"g1_{example}_still_{fraction:03d}.png")
        assert os.path.getsize(path) > 0
        width, height = _size(path)
        assert (width, height) == (
            render_module.PANEL_W,
            render_module.PANEL_H + GAP + render_module.HUD_H,
        )


def test_side_by_side_stills_are_two_panels_and_a_wide_hud(render_module, g1_model, tmp_path):
    out = tmp_path / "out"
    npz = _write(tmp_path, g1_model, "scramble")
    unfiltered = _write(tmp_path, g1_model, "scramble", unfiltered=True)
    paths = render_module.render(
        "scramble",
        npz,
        unfiltered_npz=unfiltered,
        out_dir=str(out),
        side_by_side=True,
        stills=True,
        max_seconds=0.3,
    )
    # filter on the basename: pytest's tmp_path is itself named after this test
    names = [os.path.basename(p) for p in paths]
    single = [p for p, n in zip(paths, names) if "_side_by_side" not in n]
    both = [p for p, n in zip(paths, names) if "_side_by_side" in n]
    assert len(single) == 5 and len(both) == 5
    assert _size(single[0]) == (
        render_module.PANEL_W,
        render_module.PANEL_H + GAP + render_module.HUD_H,
    )
    assert _size(both[0]) == (
        2 * render_module.SBS_PANEL_W + GAP,
        render_module.SBS_PANEL_H + GAP + render_module.SBS_HUD_H,
    )


# --------------------------------------------------------------------------- video
@needs_ffmpeg
def test_render_writes_an_mp4_only(render_module, g1_model, tmp_path):
    out = tmp_path / "out"
    paths = render_module.render(
        "navigate",
        _write(tmp_path, g1_model, "navigate"),
        out_dir=str(out),
        side_by_side=False,
        mp4=True,
        gif=False,
        max_seconds=0.2,
    )
    assert paths == [str(out / "g1_navigate.mp4")]
    assert os.path.getsize(paths[0]) > 0
    assert not (out / "g1_navigate.gif").exists()


@needs_ffmpeg
def test_render_writes_both_encodings(render_module, g1_model, tmp_path):
    out = tmp_path / "out"
    paths = render_module.render(
        "scramble",
        _write(tmp_path, g1_model, "scramble"),
        out_dir=str(out),
        side_by_side=False,
        max_seconds=0.3,
    )
    assert [os.path.basename(p) for p in paths] == ["g1_scramble.mp4", "g1_scramble.gif"]
    assert all(os.path.getsize(p) > 0 for p in paths)


@needs_ffmpeg
def test_gif_size_knobs_reach_the_output(render_module, g1_model, tmp_path):
    """--gif-width/-colors/-fps and --gif-no-hud change the GIF only; the MP4 keeps its HUD."""
    out = tmp_path / "out"
    paths = render_module.render(
        "scramble",
        _write(tmp_path, g1_model, "scramble"),
        out_dir=str(out),
        max_seconds=0.5,
        gif_width=240,
        gif_colors=32,
        gif_fps=10.0,
        gif_no_hud=True,
        no_drift=True,
    )
    mp4_info, gif_info = _probe(paths[0]), _probe(paths[1])
    # the MP4 keeps its 160 px HUD band; the GIF is the bare panel, scaled to 240 px
    assert (
        f"{render_module.PANEL_W}x{render_module.PANEL_H + GAP + render_module.HUD_H}" in mp4_info
    )
    assert f"240x{round(240 * render_module.PANEL_H / render_module.PANEL_W)}" in gif_info
    assert "10 fps" in gif_info


def test_gif_fps_default_keeps_the_20_fps_2x_contract(render_module):
    assert render_module.GIF_FPS_OUT == 20.0
    assert render_module.FPS_GIF * render_module.GIF_SPEED == render_module.GIF_FPS_OUT


@needs_ffmpeg
def test_frame_writer_gif_fps_resamples_without_changing_speed(tmp_path):
    """The writer's output rate is independent of the added-frame rate and the speed."""
    frames = [np.full((48, 64, 3), 10 * i, np.uint8) for i in range(20)]
    plain, resampled = tmp_path / "a.gif", tmp_path / "b.gif"
    for path, gif_fps in ((plain, None), (resampled, 10.0)):
        with showcase.FrameWriter(
            path, 10, kind="gif", speed=2.0, gif_width=32, gif_fps=gif_fps
        ) as w:
            for frame in frames:
                w.add(frame)
    assert "20 fps" in _probe(plain)  # fps * speed
    assert "10 fps" in _probe(resampled)
    assert resampled.stat().st_size < plain.stat().st_size


def test_no_drift_holds_the_azimuth(render_module):
    azimuths = []
    for drift in (render_module.DRIFT_DEG_PER_S, 0.0):
        cam = showcase.CameraSchedule(3.2, -18.0, azimuth0=135.0, drift_deg_per_s=drift, dt=DT)
        azimuths.append([float(cam.update(k, (0.0, 0.0)).azimuth) for k in (0, 500)])
    assert azimuths[0][1] != azimuths[0][0]  # drifts by default
    assert azimuths[1][1] == pytest.approx(azimuths[1][0])  # held with --no-drift


def test_strides_are_25_and_10_fps_at_dt_002(render_module):
    stride_mp4, fps_mp4, stride_gif, fps_gif = render_module._strides(0.02)
    assert (stride_mp4, stride_gif) == (2, 5)
    assert fps_mp4 == pytest.approx(25.0)
    assert fps_gif == pytest.approx(10.0)  # x GIF_SPEED = 20 fps of output
    assert fps_gif * render_module.GIF_SPEED == pytest.approx(20.0)


# --------------------------------------------------------------------------- the HUD edit
def test_hud_second_trace_changes_the_strip_and_its_limits():
    t_hist = np.linspace(0.0, 3.0, 60)
    h_hist = np.linspace(0.6, 0.2, 60)
    h_unfiltered = np.linspace(0.6, -0.8, 60)
    renderer = showcase.HudRenderer(640, 160)
    try:
        plain = renderer.draw(3.0, 0.2, h_hist, t_hist, 0.1, True, {})
        lo_plain = renderer._ax.get_ylim()[0]
        both = renderer.draw(3.0, 0.2, h_hist, t_hist, 0.1, True, {}, h_hist2=h_unfiltered)
        assert not np.array_equal(plain, both)
        assert renderer._ax.get_ylim()[0] < lo_plain  # the red trace widened the window
        assert len(renderer._trace2.get_xdata()) > 0
        # and it goes away again
        renderer.draw(3.0, 0.2, h_hist, t_hist, 0.1, True, {})
        assert len(renderer._trace2.get_xdata()) == 0
    finally:
        renderer.close()


def test_hud_ylim_overrides_the_per_frame_autoscale():
    t_hist = np.linspace(0.0, 3.0, 60)
    h_hist = np.linspace(6.0, 0.2, 60)
    renderer = showcase.HudRenderer(640, 160)
    try:
        renderer.draw(3.0, 0.2, h_hist, t_hist, 0.1, True, {})
        assert renderer._ax.get_ylim()[1] > 2.0  # autoscaled to the tall opening
        renderer.draw(3.0, 0.2, h_hist, t_hist, 0.1, True, {}, ylim=(-0.25, 1.5))
        assert renderer._ax.get_ylim() == pytest.approx((-0.25, 1.5))
        # a later frame with a different window keeps the same range
        renderer.draw(9.0, 0.1, h_hist, t_hist + 6.0, 0.1, True, {}, ylim=(-0.25, 1.5))
        assert renderer._ax.get_ylim() == pytest.approx((-0.25, 1.5))
    finally:
        renderer.close()


# --------------------------------------------------------------------------- floor styles
@pytest.mark.parametrize("floor", ["grid", "checker", "plain"])
def test_every_floor_style_compiles_with_a_ground_material(floor):
    try:
        model = showcase.render_model("unitree12", offline=True, floor=floor)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        pytest.skip(f"G1 render model unavailable (asset cache): {exc}")
    mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, showcase._SHOWCASE_MAT)
    assert mat >= 0
    assert model.mat_reflectance[mat] == pytest.approx(showcase.FLOOR_REFLECTANCE)
    planes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert planes and all(model.geom_matid[g] == mat for g in planes)
    tex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, showcase._SHOWCASE_TEX)
    if floor == "plain":
        assert tex < 0  # no texture at all: a flat material is the cheapest ground to encode
    else:
        assert tex >= 0
        assert model.mat_texuniform[mat]
        expected = [1.0, 1.0] if floor == "grid" else list(showcase.FLOOR_TEXREPEAT)
        assert list(model.mat_texrepeat[mat]) == pytest.approx(expected)


def test_grid_texture_is_flat_with_a_darker_border():
    n, w = showcase.GRID_TEX_SIZE, showcase.GRID_LINE_PX
    rgb = np.frombuffer(showcase._grid_texture_bytes(), dtype=np.uint8).reshape(n, n, 3)
    field = np.round(np.asarray(showcase.GRID_RGB) * 255).astype(np.uint8)
    line = np.round(np.asarray(showcase.GRID_LINE_RGB) * 255).astype(np.uint8)
    assert np.array_equal(rgb[n // 2, n // 2], field)  # uniform inside
    assert np.array_equal(rgb[0, n // 2], line) and np.array_equal(rgb[n // 2, 0], line)
    assert np.array_equal(rgb[w - 1, n // 2], line)  # the border is GRID_LINE_PX wide
    assert np.array_equal(rgb[w, n // 2], field)  # and no wider
    assert (line < field).all()  # the grid line is darker than the ground


def test_render_model_rejects_an_unknown_floor():
    with pytest.raises(ValueError, match="unknown floor"):
        showcase.render_model("unitree12", offline=True, floor="parquet")


def test_render_driver_defaults_to_the_grid_floor(render_module, g1_model, tmp_path):
    out = tmp_path / "out"
    paths = render_module.render(
        "navigate", _write(tmp_path, g1_model, "navigate"), out_dir=str(out), stills=True
    )
    assert len(paths) == 5
    assert ("unitree12", True, "grid") in render_module._MODEL_CACHE


# --------------------------------------------------------------------------- primitive look
def test_pedestrian_body_scale_widens_the_capsule(g1_model):
    scene = mujoco.MjvScene(g1_model, maxgeom=64)
    assert showcase.pedestrian(scene, (0, 0), (1, 0), 0.30, (0, 0, 1, 1), body_scale=0.8) == 3
    body, _arrow_geom, disc = scene.geoms[0], scene.geoms[1], scene.geoms[2]
    assert float(body.size[0]) == pytest.approx(0.8 * 0.30)
    assert float(disc.size[0]) == pytest.approx(0.30)  # the footprint disc keeps radius r
    scene.ngeom = 0
    showcase.pedestrian(scene, (0, 0), (1, 0), 0.30, (0, 0, 1, 1))  # the tuned default
    assert float(scene.geoms[0].size[0]) == pytest.approx(0.8 * 0.30)


def test_beacon_pole_is_a_hairline(g1_model):
    scene = mujoco.MjvScene(g1_model, maxgeom=64)
    assert showcase.beacon(scene, (1.0, 2.0), 0.12, (0.1, 0.8, 0.2, 0.6)) == 2
    sphere, pole = scene.geoms[0], scene.geoms[1]
    assert float(sphere.size[0]) == pytest.approx(0.12)
    assert float(pole.size[0]) == pytest.approx(showcase.BEACON_POLE_RADIUS)
    assert float(pole.size[2]) == pytest.approx(showcase.BEACON_POLE_HEIGHT / 2.0)
    assert float(pole.rgba[3]) == pytest.approx(0.5 * 0.6)


def test_hud_bar_full_scale_stops_the_acceleration_bar_pegging():
    renderer = showcase.HudRenderer(640, 160)
    try:
        renderer.draw(1.0, 0.3, None, None, 0.9, True, {})  # 0.9 m/s^2 on a 0.5 scale: pegged
        assert renderer._bar.get_width() == pytest.approx(0.5)
        renderer.draw(1.0, 0.3, None, None, 0.9, True, {}, full_scale=2.0)
        assert renderer._bar.get_width() == pytest.approx(0.9)
        assert renderer._bar_ax.get_xlim() == pytest.approx((0.0, 2.0))
        assert renderer._bar_bg.get_width() == pytest.approx(2.0)
    finally:
        renderer.close()


def test_hud_pills_shrink_but_never_disappear():
    renderer = showcase.HudRenderer(1280, 160)
    try:
        many = [
            ("CBF ACTIVE", "#0f0"),
            ("MPPI FALLBACK", "#fa0"),
            ("QP NOT CONVERGED", "#fa0"),
            ("SLACK 0.0842", "#f00"),
            ("EXTRA FLAG HERE", "#fa0"),
        ]
        renderer._set_pills(many)
        assert len([p for p in renderer._pills if p.get_visible()]) == len(many)
        with pytest.raises(ValueError, match="only .* slots"):
            renderer._set_pills(many + [("ONE TOO MANY", "#fa0")])
    finally:
        renderer.close()


def test_hud_cache_is_bounded_and_closes_what_it_evicts():
    showcase._HUD_CACHE.clear()
    sizes = [(320, 120), (360, 120), (400, 120)]
    for width, height in sizes:
        showcase.hud_strip(width, height, 0.0, 0.0, None, None, 0.0, False, {})
    assert len(showcase._HUD_CACHE) <= showcase._HUD_CACHE_MAX
    assert sizes[0] not in showcase._HUD_CACHE  # the oldest was evicted, not kept forever
    assert sizes[-1] in showcase._HUD_CACHE
    showcase._HUD_CACHE.clear()


def test_frame_writer_removes_a_half_written_file_on_error(tmp_path):
    out = tmp_path / "aborted.mp4"
    with pytest.raises(RuntimeError, match="boom"):
        with showcase.FrameWriter(out, 25) as w:
            w.add(np.zeros((48, 64, 3), np.uint8))
            raise RuntimeError("boom")
    assert not out.exists()  # a truncated clip must not survive to be read as a result


def test_hud_intervention_caption_is_overridable():
    renderer = showcase.HudRenderer(640, 160)
    try:
        renderer.draw(1.0, 0.3, None, None, 0.2, True, {})
        assert "m/s" in renderer._bar_caption.get_text()
        renderer.draw(
            1.0, 0.3, None, None, 0.2, True, {}, intervention_caption="intervention  |da| = 0.20"
        )
        assert renderer._bar_caption.get_text() == "intervention  |da| = 0.20"
    finally:
        renderer.close()
