"""Unit tests for ``cbfkit.systems.mujoco.showcase``.

The whole module is skipped without MuJoCo. Nothing here needs the G1 asset cache: the
scene tests build a tiny inline model, and the one test that does build a real showcase
model is skipped when the cache is absent.
"""

import shutil
import subprocess

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from cbfkit.systems.mujoco import showcase  # noqa: E402

_TINY_XML = """
<mujoco model="tiny">
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0 0 0" width="4" height="4"/>
    <texture type="2d" name="src_ground" builtin="checker" rgb1="1 0 0" rgb2="0 1 0"
      width="32" height="32"/>
    <material name="src_ground" texture="src_ground"/>
  </asset>
  <worldbody>
    <light name="src_light" pos="1 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="src_ground"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def tiny_model():
    return mujoco.MjModel.from_xml_string(_TINY_XML)


@pytest.fixture
def scene(tiny_model):
    """An empty ``MjvScene`` with room for overlay geoms."""
    return mujoco.MjvScene(tiny_model, maxgeom=512)


def _has_ffmpeg() -> bool:
    try:
        showcase.ffmpeg_exe()
    except RuntimeError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="no ffmpeg available")


# --------------------------------------------------------------------------- primitives
def test_disc_adds_one_geom(scene):
    assert showcase.disc(scene, (1.0, 2.0), 0.5, (1, 0, 0, 0.35)) == 1
    assert scene.ngeom == 1
    geom = scene.geoms[0]
    assert geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert geom.pos[2] == pytest.approx(0.004)
    assert geom.size[0] == pytest.approx(0.5)


def test_ring_adds_one_geom_per_segment(scene):
    assert showcase.ring(scene, (0.0, 0.0), 1.0, (0, 1, 0, 0.5), n=12) == 12
    assert scene.ngeom == 12
    assert all(scene.geoms[i].type == mujoco.mjtGeom.mjGEOM_CAPSULE for i in range(12))
    # every segment endpoint sits on the circle at the requested height
    for i in range(12):
        assert np.linalg.norm(scene.geoms[i].pos[:2]) < 1.0 + 1e-9
        assert scene.geoms[i].pos[2] == pytest.approx(0.006)


def test_arrow_adds_one_geom_and_skips_degenerate(scene):
    assert showcase.arrow(scene, (0, 0, 1), (1, 0, 1), (1, 1, 1, 1)) == 1
    assert scene.geoms[0].type == mujoco.mjtGeom.mjGEOM_ARROW
    assert showcase.arrow(scene, (0, 0, 1), (0, 0, 1), (1, 1, 1, 1)) == 0
    assert scene.ngeom == 1


def test_ellipse_mat_is_the_z_rotation(scene):
    theta = 0.7
    assert showcase.ellipse(scene, (1.0, -1.0), 0.3, 0.5, theta, (1, 1, 0, 0.4)) == 1
    geom = scene.geoms[0]
    assert geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
    c, s = np.cos(theta), np.sin(theta)
    expected = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(np.asarray(geom.mat).reshape(3, 3), expected, atol=1e-12)
    assert geom.size[0] == pytest.approx(0.3)
    assert geom.size[1] == pytest.approx(0.5)
    assert geom.size[2] == pytest.approx(0.002)


def test_path_adds_one_geom_per_segment(scene):
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    assert showcase.path(scene, pts, (1, 1, 1, 1)) == 3
    assert scene.ngeom == 3
    assert showcase.path(scene, pts[:1], (1, 1, 1, 1)) == 0


def test_trail_ramps_alpha_along_the_path(scene):
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert showcase.trail(scene, pts, (0.1, 0.2, 0.3, 0.8)) == 3
    alphas = [float(scene.geoms[i].rgba[3]) for i in range(3)]
    assert alphas == pytest.approx([0.8 / 3.0, 2 * 0.8 / 3.0, 0.8])
    assert alphas[0] < alphas[-1]


def test_pedestrian_geom_count(scene):
    # body + heading arrow + floor disc
    assert showcase.pedestrian(scene, (1.0, 1.0), (0.8, 0.0), 0.3, (0.2, 0.4, 0.9, 1.0)) == 3
    scene.ngeom = 0
    # standing still: no arrow
    assert showcase.pedestrian(scene, (1.0, 1.0), (0.0, 0.0), 0.3, (0.2, 0.4, 0.9, 1.0)) == 2


def test_pedestrian_arrow_length_is_clipped(scene):
    showcase.pedestrian(scene, (0.0, 0.0), (10.0, 0.0), 0.3, (1, 1, 1, 1))
    arrow_geom = scene.geoms[1]
    assert arrow_geom.type == mujoco.mjtGeom.mjGEOM_ARROW
    # mjv_connector puts the segment length in size[2]; 0.5 * |v| = 5 m is clipped to 0.6
    assert float(arrow_geom.size[2]) == pytest.approx(0.6)


def test_beacon_adds_sphere_and_column(scene):
    assert showcase.beacon(scene, (2.0, 3.0), 0.25, (0.1, 0.8, 0.2, 0.6)) == 2
    assert scene.geoms[0].type == mujoco.mjtGeom.mjGEOM_SPHERE
    assert scene.geoms[0].pos[2] == pytest.approx(0.25)
    assert scene.geoms[1].type == mujoco.mjtGeom.mjGEOM_CYLINDER


def test_primitives_return_zero_when_the_scene_is_full(tiny_model):
    full = mujoco.MjvScene(tiny_model, maxgeom=1)
    assert showcase.disc(full, (0, 0), 1.0, (1, 1, 1, 1)) == 1
    assert full.ngeom == full.maxgeom
    assert showcase.disc(full, (0, 0), 1.0, (1, 1, 1, 1)) == 0
    assert showcase.ring(full, (0, 0), 1.0, (1, 1, 1, 1)) == 0
    assert showcase.arrow(full, (0, 0, 0), (1, 0, 0), (1, 1, 1, 1)) == 0
    assert showcase.ellipse(full, (0, 0), 1.0, 1.0, 0.0, (1, 1, 1, 1)) == 0
    assert showcase.path(full, np.zeros((3, 2)) + np.arange(3)[:, None], (1, 1, 1, 1)) == 0
    assert showcase.beacon(full, (0, 0), 0.2, (1, 1, 1, 1)) == 0
    assert showcase.pedestrian(full, (0, 0), (1, 0), 0.3, (1, 1, 1, 1)) == 0


# --------------------------------------------------------------------------- colours
def test_h_rgba_endpoints():
    green = showcase.h_rgba(0.5)
    assert np.allclose(green[:3], (0.20, 0.75, 0.35), atol=1e-6)
    assert np.allclose(showcase.h_rgba(3.0)[:3], green[:3])
    amber = showcase.h_rgba(0.0)
    assert np.allclose(amber[:3], (0.95, 0.65, 0.10), atol=1e-6)
    red = showcase.h_rgba(-0.5)
    assert np.allclose(red[:3], (0.85, 0.15, 0.20), atol=1e-6)
    assert np.allclose(showcase.h_rgba(-4.0)[:3], red[:3])


def test_h_rgba_interpolates_and_carries_alpha():
    mid = showcase.h_rgba(0.25, alpha=0.9)
    assert mid[3] == pytest.approx(0.9)
    amber, green = showcase.h_rgba(0.0)[:3], showcase.h_rgba(0.5)[:3]
    assert np.allclose(mid[:3], 0.5 * (amber + green), atol=1e-6)
    assert showcase.h_rgba(0.0)[3] == pytest.approx(0.35)  # default alpha
    assert showcase.h_rgba(1.0).dtype == np.float32


# --------------------------------------------------------------------------- camera
def test_camera_snaps_on_the_first_frame_then_lags():
    cam_sched = showcase.CameraSchedule(3.6, -18.0, azimuth0=100.0, lag_s=0.4, dt=0.02)
    cam = cam_sched.update(0, (5.0, -2.0))
    assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert cam.lookat[0] == pytest.approx(5.0)
    assert cam.lookat[1] == pytest.approx(-2.0)
    assert cam.lookat[2] == pytest.approx(0.8)
    assert cam.distance == pytest.approx(3.6)
    assert cam.elevation == pytest.approx(-18.0)
    # one step later the look-at has moved only part of the way to the new target
    cam = cam_sched.update(1, (6.0, -2.0))
    alpha = 1.0 - np.exp(-0.02 / 0.4)
    assert cam.lookat[0] == pytest.approx(5.0 + alpha * 1.0)
    assert 0.0 < alpha < 0.1


def test_camera_lag_accounts_for_skipped_steps():
    a = showcase.CameraSchedule(4.0, -20.0, lag_s=0.4, dt=0.02)
    b = showcase.CameraSchedule(4.0, -20.0, lag_s=0.4, dt=0.02)
    a.update(0, (0.0, 0.0))
    b.update(0, (0.0, 0.0))
    # stepping by 2 at once must move as far as the 2-step limit of the same lag
    x_two = float(a.update(2, (1.0, 0.0)).lookat[0])
    b.update(1, (1.0, 0.0))
    x_one_one = float(b.update(2, (1.0, 0.0)).lookat[0])
    assert x_two == pytest.approx(x_one_one, abs=1e-12)


def test_camera_azimuth_drifts_and_respects_heading():
    sched = showcase.CameraSchedule(4.0, -20.0, azimuth0=135.0, drift_deg_per_s=6.0, dt=0.02)
    sched.update(0, (0.0, 0.0))
    assert sched.update(0, (0.0, 0.0)).azimuth == pytest.approx(135.0)
    az = float(sched.update(50, (0.0, 0.0)).azimuth)  # t = 1 s
    assert az == pytest.approx(141.0)
    # heading 0 puts the camera behind-left at 145 deg; blended 50/50 with the drift
    sched2 = showcase.CameraSchedule(4.0, -20.0, azimuth0=135.0, drift_deg_per_s=0.0, dt=0.02)
    with_heading = float(sched2.update(0, (0.0, 0.0), heading=0.0).azimuth)
    assert with_heading == pytest.approx(0.5 * (135.0 + 145.0))


# --------------------------------------------------------------------------- HUD
def test_hud_strip_shape_and_band_colour():
    out = showcase.hud_strip(
        640,
        160,
        3.2,
        0.41,
        np.linspace(0.8, 0.41, 100),
        np.linspace(0.0, 3.2, 100),
        0.12,
        True,
        {"mppi fallback": False, "qp fail": True},
        slack=0.004,
        mode="CBF-QP + MPPI",
    )
    assert out.shape == (160, 640, 3)
    assert out.dtype == np.uint8
    corner = out[0, 0]  # the band colour, pre-composited
    assert np.allclose(corner, np.round(np.array(showcase._BAND_RGB) * 255.0), atol=2)
    assert out.std() > 1.0  # something was actually drawn


def test_hud_strip_handles_empty_history_and_caches_the_figure():
    a = showcase.hud_strip(320, 120, 0.0, 0.0, None, None, 0.0, False, {})
    assert a.shape == (120, 320, 3)
    key = (320, 120)
    assert key in showcase._HUD_CACHE
    renderer = showcase._HUD_CACHE[key]
    b = showcase.hud_strip(320, 120, 0.0, 0.0, np.array([]), np.array([]), 0.0, False, {})
    assert showcase._HUD_CACHE[key] is renderer  # reused, not rebuilt
    assert b.shape == a.shape


def test_hud_pills_wrap_to_a_second_row_instead_of_being_dropped():
    renderer = showcase.HudRenderer(1280, 160)
    try:
        four = [
            ("CBF ACTIVE", "#0f0"),
            ("MPPI FB", "#fa0"),
            ("QP FAIL", "#fa0"),
            ("SLACK 0.084", "#f00"),
        ]
        rows = renderer._pack_pills(four, renderer._mid)
        assert len(rows) == 2
        assert sum(len(r) for r in rows) == 4  # nothing dropped
        assert rows[0][0] == pytest.approx(showcase._PILL_X0)
        assert rows[1][0] == pytest.approx(showcase._PILL_X0)
        renderer._set_pills(four)
        visible = [p for p in renderer._pills if p.get_visible()]
        assert len(visible) == 4
        ys = {round(float(p.get_position()[1]), 3) for p in visible}
        assert ys == {round(y, 3) for y in showcase._PILL_Y_ROWS}
        # one pill stays on the single bottom row
        renderer._set_pills(four[:1])
        single = [p for p in renderer._pills if p.get_visible()]
        assert len(single) == 1
        assert float(single[0].get_position()[1]) == pytest.approx(showcase._PILL_Y_SINGLE)
    finally:
        renderer.close()


def test_hud_strip_negative_h_changes_the_readout():
    args = (480, 140, 2.0)
    hist_t, hist_h = np.linspace(0.0, 2.0, 50), np.linspace(0.3, -0.2, 50)
    safe = showcase.hud_strip(*args, 0.3, hist_h, hist_t, 0.0, False, {})
    unsafe = showcase.hud_strip(*args, -0.2, hist_h, hist_t, 0.3, True, {})
    assert not np.array_equal(safe, unsafe)


# --------------------------------------------------------------------------- compositing
def test_compose_panels_size_arithmetic():
    left = np.full((540, 960, 3), 30, np.uint8)
    right = np.full((540, 960, 3), 60, np.uint8)
    hud = np.full((180, 1920, 3), 40, np.uint8)
    out = showcase.compose_panels([left, right], ["policy only", "+ CBF filter"], hud, gap=8)
    assert out.shape == (540 + 8 + 180, 960 * 2 + 8, 3)
    assert out.dtype == np.uint8
    # the gap column between the panels carries the background colour
    assert tuple(out[300, 962]) == (18, 18, 20)


def test_compose_panels_single_panel_without_hud():
    panel = np.full((360, 640, 3), 50, np.uint8)
    out = showcase.compose_panels([panel], ["CBF"], None)
    assert out.shape == (360, 640, 3)
    # the label pill darkened the top-left corner region
    assert out[20, 40].mean() != pytest.approx(50.0, abs=0.5)


def test_compose_panels_centres_the_narrower_row():
    panel = np.full((200, 300, 3), 90, np.uint8)
    hud = np.full((60, 500, 3), 20, np.uint8)
    out = showcase.compose_panels([panel], ["x"], hud, gap=4)
    assert out.shape == (200 + 4 + 60, 500, 3)
    assert tuple(out[100, 5]) == (18, 18, 20)  # left margin beside the centred panel
    assert tuple(out[100, 250]) == (90, 90, 90)


def test_compose_panels_rejects_mismatched_labels():
    panel = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError):
        showcase.compose_panels([panel, panel], ["only one"], None)
    with pytest.raises(ValueError):
        showcase.compose_panels([], [], None)


# --------------------------------------------------------------------------- writers
def _frames(n=5, w=64, h=48):
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, :, 0] = 40 + 30 * i
        frame[: h // 2, : w // 2, 1] = 200
        yield frame


def _probe(path):
    """``ffmpeg -i <path>`` output (ffmpeg prints the stream table on stderr and exits 1)."""
    out = subprocess.run(
        [showcase.ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True
    )
    return out.stderr.decode(errors="replace")


@needs_ffmpeg
def test_frame_writer_mp4(tmp_path):
    out = tmp_path / "clip.mp4"
    writer = showcase.FrameWriter(out, 25, kind="mp4")
    for frame in _frames():
        writer.add(frame)
    assert writer.close() == str(out)
    assert writer.n_frames == 5
    assert out.stat().st_size > 0
    info = _probe(out)
    assert "Video: h264" in info
    assert "64x48" in info


@needs_ffmpeg
def test_frame_writer_gif(tmp_path):
    out = tmp_path / "clip.gif"
    with showcase.FrameWriter(out, 10, kind="gif", speed=2.0, gif_width=32, gif_colors=32) as w:
        for frame in _frames():
            w.add(frame)
    assert out.stat().st_size > 0
    info = _probe(out)
    assert "Video: gif" in info
    assert "32x24" in info


@needs_ffmpeg
def test_frame_writer_pads_odd_sizes_and_rejects_changes(tmp_path):
    out = tmp_path / "odd.mp4"
    writer = showcase.FrameWriter(out, 25)
    writer.add(np.zeros((45, 63, 3), np.uint8))
    writer.add(np.zeros((45, 63, 3), np.uint8))
    with pytest.raises(ValueError):
        writer.add(np.zeros((40, 60, 3), np.uint8))
    writer.close()
    assert "64x46" in _probe(out)


def test_frame_writer_validates_arguments(tmp_path):
    with pytest.raises(ValueError):
        showcase.FrameWriter(tmp_path / "x.webm", 25, kind="webm")
    writer = showcase.FrameWriter(tmp_path / "x.mp4", 25)
    with pytest.raises(RuntimeError):
        writer.close()  # nothing was added


@needs_ffmpeg
def test_frame_writer_rejects_non_rgb_frames(tmp_path):
    writer = showcase.FrameWriter(tmp_path / "x.mp4", 25)
    with pytest.raises(ValueError):
        writer.add(np.zeros((10, 10), np.uint8))


def test_ffmpeg_exe_finds_a_binary():
    if not _has_ffmpeg():
        pytest.skip("no ffmpeg available")
    exe = showcase.ffmpeg_exe()
    assert exe and (shutil.which(exe) or exe)


# --------------------------------------------------------------------------- kinematics
def test_body_positions_and_pelvis_yaw(tiny_model):
    nq = tiny_model.nq
    states = np.zeros((4, nq + tiny_model.nv + 3))
    yaws = np.array([0.0, 0.5, -1.2, 3.0])
    for k, yaw in enumerate(yaws):
        states[k, 0] = float(k)  # ball x
        states[k, 2] = 1.0
        states[k, 3] = np.cos(yaw / 2.0)  # w
        states[k, 6] = np.sin(yaw / 2.0)  # z
    ball = mujoco.mj_name2id(tiny_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    pos = showcase.body_positions(tiny_model, states, nq, [ball])
    assert pos.shape == (4, 1, 3)
    assert pos[:, 0, 0] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert pos[:, 0, 2] == pytest.approx([1.0] * 4)
    assert showcase.pelvis_yaw(states, nq) == pytest.approx(yaws)


def test_pelvis_yaw_needs_a_free_joint():
    with pytest.raises(ValueError):
        showcase.pelvis_yaw(np.zeros((2, 10)), 6)


# --------------------------------------------------------------------------- scene
def test_source_xml_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        showcase.source_xml("g1_42dof")


def test_render_model_unitree12_applies_the_showcase_look(monkeypatch):
    """Only runs against a populated asset cache; never downloads."""
    monkeypatch.setenv("CBFKIT_ASSETS_OFFLINE", "1")
    try:
        src = showcase.source_xml("unitree12", offline=True)
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip(f"G1 asset cache unavailable: {exc}")
    if not src.is_file():
        pytest.skip(f"G1 asset cache unavailable: {src} missing")
    reference = mujoco.MjModel.from_xml_path(str(src))
    model = showcase.render_model("unitree12", offline=True)
    assert (model.nq, model.nv) == (reference.nq, reference.nv)
    assert model.nlight == 2  # the source's single light was replaced by key + fill
    assert bool(model.light_castshadow[0]) and not bool(model.light_castshadow[1])
    assert model.vis.global_.offwidth == showcase.OFFWIDTH
    assert model.vis.global_.offheight == showcase.OFFHEIGHT
    assert model.vis.quality.shadowsize == showcase.SHADOWSIZE
    assert model.vis.headlight.diffuse == pytest.approx(showcase.HEADLIGHT_DIFFUSE)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    matid = int(model.geom_matid[floor])
    assert matid >= 0
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, matid) == showcase._SHOWCASE_MAT
    assert model.mat_reflectance[matid] == pytest.approx(showcase.FLOOR_REFLECTANCE)
    # a render-only edit: the logged states still replay through this model
    data = mujoco.MjData(model)
    data.qpos[:] = np.zeros(model.nq)
    mujoco.mj_kinematics(model, data)
