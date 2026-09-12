"""Presentation-quality rendering helpers for the MuJoCo G1 examples.

This module is the reusable half of the "showcase" renders: it knows how to build a
*render-only* model with studio lighting, how to draw certificate overlays into an
``mjvScene``, how to move a camera smoothly, how to draw the HUD strip, and how to
write MP4/GIF files. It holds no example-specific geometry -- that lives in
``examples/mujoco/g1_showcase.py``.

Design constraints that shaped the API:

* The markers callback used by :mod:`cbfkit.systems.mujoco.viewer_utils` receives only
  ``(scn, k, t)`` -- no ``MjData`` and no camera -- so anything derived from body poses
  must be precomputed per logged step (:func:`body_positions`, :func:`pelvis_yaw`) and
  the camera must be driven by a custom render loop (:class:`CameraSchedule`).
* ``viewer_utils.add_marker`` hardcodes identity rotation, so every oriented primitive
  here calls ``mujoco.mjv_initGeom`` (and ``mujoco.mjv_connector``) directly.
* Everything is numpy in / numpy out. No JAX, and matplotlib is imported only inside
  the HUD renderer so importing this module stays cheap.

Overlay height convention: the ``z`` argument of every floor primitive is the height of
the geom *centre* above the floor, and the defaults stack in a fixed order so overlays
never z-fight: disc 4 mm, ellipse 5 mm, ring 6 mm, trail 8 mm, path 10 mm.
"""

import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mujoco
import numpy as np

# Anything acceptable as a small numeric vector (positions, sizes, colours).
Vec = Union[Sequence[float], np.ndarray]

__all__ = [
    "PLANT_KINDS",
    "render_model",
    "source_xml",
    "h_rgba",
    "disc",
    "ring",
    "arrow",
    "ellipse",
    "path",
    "trail",
    "pedestrian",
    "beacon",
    "CameraSchedule",
    "HudRenderer",
    "hud_strip",
    "compose_panels",
    "FrameWriter",
    "body_positions",
    "pelvis_yaw",
]

PLANT_KINDS: Tuple[str, ...] = ("unitree12", "amo23", "groot29")

# --------------------------------------------------------------------------- scene look
# The tuned reference is ``models/g1/scene_showcase.xml`` ("studio" variant, chosen on the
# render box 2026-09-12). These constants mirror it so every plant gets the same look.
FLOOR_RGB1 = (0.52, 0.53, 0.56)
FLOOR_RGB2 = (0.46, 0.47, 0.50)
FLOOR_MARKRGB = (0.60, 0.61, 0.64)
FLOOR_TEXREPEAT = (8.0, 8.0)
FLOOR_REFLECTANCE = 0.08

# Floor styles. The fine checker is the single biggest contributor to GIF size: under a
# tracking camera every one of its edges moves, so no two frames share a background and the
# palette encoder has nothing to reuse. Measured on the scramble clip at 400 px / 48 colours:
# checker 11 MB, grid 5.1 MB, plain 3.4 MB.
FLOOR_STYLES: Tuple[str, ...] = ("grid", "checker", "plain")
GRID_RGB = (0.51, 0.52, 0.53)
GRID_LINE_RGB = (0.42, 0.42, 0.42)
GRID_TEX_SIZE = 300  # one tile is 1 m (texuniform + texrepeat 1 1)
GRID_LINE_PX = 4
SKY_RGB1 = (0.70, 0.78, 0.90)
SKY_RGB2 = (0.95, 0.97, 1.00)
HAZE_RGBA = (0.85, 0.87, 0.90, 1.0)
HEADLIGHT_DIFFUSE = (0.25, 0.25, 0.25)
HEADLIGHT_AMBIENT = (0.15, 0.15, 0.15)
HEADLIGHT_SPECULAR = (0.20, 0.20, 0.20)
KEY_LIGHT = dict(
    pos=(3.0, -4.0, 6.0),
    dir=(-0.4, 0.55, -0.75),
    diffuse=(0.70, 0.70, 0.68),
    specular=(0.30, 0.30, 0.30),
    ambient=(0.0, 0.0, 0.0),
)
FILL_LIGHT = dict(
    pos=(-4.0, 3.0, 4.0),
    dir=(0.6, -0.45, -0.65),
    diffuse=(0.20, 0.21, 0.24),
    specular=(0.05, 0.05, 0.05),
    ambient=(0.0, 0.0, 0.0),
)
SHADOWSIZE = 8192
OFFSAMPLES = 8
SHADOWCLIP = 10.0  # the shadow-frustum edge shows as a moire wedge on the floor below ~8
SHADOWSCALE = 0.5
OFFWIDTH = 1920
OFFHEIGHT = 1080

_SHOWCASE_TEX = "cbfkit_showcase_ground"
_SHOWCASE_MAT = "cbfkit_showcase_ground"


def source_xml(plant_kind: str, *, offline: bool = False) -> Path:
    """Path to the MJCF a plant kind is built from (the file the showcase look is applied to)."""
    if plant_kind not in PLANT_KINDS:
        raise ValueError(f"unknown plant_kind {plant_kind!r}; expected one of {PLANT_KINDS}")
    from cbfkit.systems.mujoco import assets

    if plant_kind == "unitree12":
        return (
            assets.unitree_rl_gym_dir(offline=offline) / "resources/robots/g1_description/scene.xml"
        )
    if plant_kind == "amo23":
        return assets.amo_dir(offline=offline) / "g1.xml"
    # GR00T ships a robot-only model; ``load_g1_29dof`` writes the patched sibling that has
    # the injected ground plane. Call it once so that file exists, then render from it.
    from cbfkit.systems.mujoco import groot_policy

    groot_policy.load_g1_29dof(offline=offline)
    src = assets.groot_dir(offline=offline) / groot_policy._REL_XML
    return src.with_name("g1_29dof_floor.xml")


def _clear_lights(spec: mujoco.MjSpec) -> None:
    for light in list(spec.lights):
        spec.delete(light)


def _clear_skyboxes(spec: mujoco.MjSpec) -> None:
    for tex in list(spec.textures):
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(tex)


def _grid_texture_bytes() -> bytes:
    """A ``GRID_TEX_SIZE`` square of :data:`GRID_RGB` with a darker border of ``GRID_LINE_PX``.

    Tiled uniformly at one tile per metre this reads as a 1 m grid. MuJoCo's builtin marks
    cannot set a border width, so the pixels are written directly.
    """
    n, w = int(GRID_TEX_SIZE), int(GRID_LINE_PX)
    rgb = np.empty((n, n, 3), dtype=np.uint8)
    rgb[:] = np.round(np.asarray(GRID_RGB) * 255.0).astype(np.uint8)
    line = np.round(np.asarray(GRID_LINE_RGB) * 255.0).astype(np.uint8)
    rgb[:w, :, :] = line
    rgb[-w:, :, :] = line
    rgb[:, :w, :] = line
    rgb[:, -w:, :] = line
    return rgb.tobytes()


def _add_showcase_floor(spec: mujoco.MjSpec, floor: str) -> None:
    """Add the showcase ground material and point every plane geom at it.

    Adding a *new* texture/material (rather than editing the source's) is what makes this
    work uniformly for GR00T, whose injected floor has no material at all. ``floor`` is one
    of :data:`FLOOR_STYLES`: ``grid`` (flat with 1 m lines, the default), ``checker`` (the
    finer original) or ``plain`` (no texture at all). The material's reflectance is the same
    for all three, so only the albedo pattern changes.
    """
    if floor not in FLOOR_STYLES:
        raise ValueError(f"unknown floor {floor!r}; expected one of {FLOOR_STYLES}")
    mat = spec.add_material(name=_SHOWCASE_MAT)
    mat.reflectance = FLOOR_REFLECTANCE
    if floor == "plain":
        mat.rgba = [*GRID_RGB, 1.0]
    elif floor == "checker":
        spec.add_texture(
            name=_SHOWCASE_TEX,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            rgb1=list(FLOOR_RGB1),
            rgb2=list(FLOOR_RGB2),
            markrgb=list(FLOOR_MARKRGB),
            width=300,
            height=300,
        )
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _SHOWCASE_TEX
        mat.texuniform = True
        mat.texrepeat = list(FLOOR_TEXREPEAT)
    else:  # grid
        texture = spec.add_texture(
            name=_SHOWCASE_TEX,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_NONE,
            width=int(GRID_TEX_SIZE),
            height=int(GRID_TEX_SIZE),
            nchannel=3,
        )
        texture.data = _grid_texture_bytes()
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _SHOWCASE_TEX
        mat.texuniform = True
        mat.texrepeat = [1.0, 1.0]  # one tile per metre
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.material = _SHOWCASE_MAT


def _add_showcase_sky(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=list(SKY_RGB1),
        rgb2=list(SKY_RGB2),
        width=512,
        height=3072,
    )


def _add_showcase_lights(spec: mujoco.MjSpec) -> None:
    directional = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    for name, cfg, castshadow in (
        ("showcase_key", KEY_LIGHT, True),
        ("showcase_fill", FILL_LIGHT, False),
    ):
        light = spec.worldbody.add_light(name=name)
        light.type = directional
        light.pos = list(cfg["pos"])
        light.dir = list(cfg["dir"])
        light.diffuse = list(cfg["diffuse"])
        light.specular = list(cfg["specular"])
        light.ambient = list(cfg["ambient"])
        light.castshadow = castshadow
        light.active = True


def _set_showcase_visual(spec: mujoco.MjSpec) -> None:
    spec.visual.headlight.diffuse = list(HEADLIGHT_DIFFUSE)
    spec.visual.headlight.ambient = list(HEADLIGHT_AMBIENT)
    spec.visual.headlight.specular = list(HEADLIGHT_SPECULAR)
    spec.visual.rgba.haze = list(HAZE_RGBA)
    spec.visual.quality.shadowsize = SHADOWSIZE
    spec.visual.quality.offsamples = OFFSAMPLES
    spec.visual.map.shadowclip = SHADOWCLIP
    spec.visual.map.shadowscale = SHADOWSCALE
    spec.visual.global_.offwidth = OFFWIDTH
    spec.visual.global_.offheight = OFFHEIGHT


def render_model(plant_kind: str, *, offline: bool = False, floor: str = "grid") -> mujoco.MjModel:
    """Compile a render-only ``MjModel`` for ``plant_kind`` with the showcase studio look.

    The returned model has the same ``nq``/``nv`` as the plant's own model (asserted), so
    logged flat states can be replayed through it verbatim. Only the worldbody's lights,
    floor material and ``<visual>`` settings differ, none of which touch the physics that
    produced the log.

    ``plant_kind`` is one of :data:`PLANT_KINDS`. With ``offline=True`` (or
    ``CBFKIT_ASSETS_OFFLINE=1``) a missing asset cache raises instead of downloading.
    ``floor`` is one of :data:`FLOOR_STYLES`; the ``grid`` default is both cleaner and much
    cheaper to encode as a GIF than the original fine ``checker``.
    """
    if floor not in FLOOR_STYLES:  # checked before any asset work, so a typo fails fast
        raise ValueError(f"unknown floor {floor!r}; expected one of {FLOOR_STYLES}")
    src = source_xml(plant_kind, offline=offline)
    # One parse, two compiles: parsing the G1 pulls in 51 meshes, so the pristine model that
    # establishes the nq/nv contract is compiled from this same spec rather than re-read.
    spec = mujoco.MjSpec.from_file(str(src))
    reference = spec.compile()
    source_layout = (reference.nq, reference.nv)
    _clear_lights(spec)
    _clear_skyboxes(spec)
    _add_showcase_sky(spec)
    _add_showcase_floor(spec, floor)
    _add_showcase_lights(spec)
    _set_showcase_visual(spec)
    model = spec.compile()
    if (model.nq, model.nv) != source_layout:
        raise RuntimeError(
            f"showcase model for {plant_kind!r} changed the state layout: "
            f"nq/nv {(model.nq, model.nv)} != source {source_layout}"
        )
    return model


# --------------------------------------------------------------------------- colours
def _rgb(values: Vec) -> Tuple[float, float, float]:
    """A plain 3-tuple of floats (what matplotlib's colour setters accept)."""
    v = np.asarray(values, dtype=float).reshape(-1)
    return (float(v[0]), float(v[1]), float(v[2]))


_GREEN = np.array([0.20, 0.75, 0.35])
_AMBER = np.array([0.95, 0.65, 0.10])
_RED = np.array([0.85, 0.15, 0.20])


def h_rgba(h: float, alpha: float = 0.35) -> np.ndarray:
    """Barrier-value colour: green at ``h >= 0.5``, amber at ``h = 0``, red at ``h <= -0.5``.

    Returns a ``(4,)`` float32 RGBA array ready for ``mjv_initGeom``.
    """
    value = float(h)
    if value >= 0.0:
        frac = min(value / 0.5, 1.0)
        rgb = _AMBER + frac * (_GREEN - _AMBER)
    else:
        frac = min(-value / 0.5, 1.0)
        rgb = _AMBER + frac * (_RED - _AMBER)
    return np.asarray([*rgb, float(alpha)], dtype=np.float32)


# --------------------------------------------------------------------------- primitives
def _init_geom(
    scn: mujoco.MjvScene,
    geom_type: int,
    size: Vec,
    pos: Vec,
    mat: np.ndarray,
    rgba: Vec,
) -> Optional[mujoco.MjvGeom]:
    """Append a visual-only geom with an explicit rotation; None when the scene is full."""
    if scn.ngeom >= scn.maxgeom:
        return None
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        int(geom_type),
        np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float),
        np.asarray(mat, dtype=float).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1
    return geom


def _connector(
    scn: mujoco.MjvScene,
    geom_type: int,
    radius: float,
    p_from: Vec,
    p_to: Vec,
    rgba: Vec,
) -> int:
    """Append one connector geom (capsule/cylinder/arrow) between two 3-D points."""
    a = np.asarray(p_from, dtype=float)
    b = np.asarray(p_to, dtype=float)
    if float(np.linalg.norm(b - a)) < 1e-9:
        return 0
    geom = _init_geom(scn, geom_type, (radius, radius, radius), a, np.eye(3), rgba)
    if geom is None:
        return 0
    mujoco.mjv_connector(geom, int(geom_type), float(radius), a, b)
    return 1


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(float(theta)), np.sin(float(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def disc(
    scn: mujoco.MjvScene,
    pos_xy: Vec,
    r: float,
    rgba: Vec,
    z: float = 0.004,
) -> int:
    """Flat 4 mm floor disc of radius ``r`` centred ``z`` above the floor."""
    p = np.asarray(pos_xy, dtype=float)
    geom = _init_geom(
        scn,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (float(r), 0.002, 0.0),
        (p[0], p[1], float(z)),
        np.eye(3),
        rgba,
    )
    return 0 if geom is None else 1


def ring(
    scn: mujoco.MjvScene,
    pos_xy: Vec,
    r: float,
    rgba: Vec,
    width: float = 0.02,
    n: int = 48,
    z: float = 0.006,
) -> int:
    """Keep-out ring of radius ``r`` as an ``n``-segment capsule polygon (MuJoCo has no torus)."""
    p = np.asarray(pos_xy, dtype=float)
    ang = np.linspace(0.0, 2.0 * np.pi, int(n) + 1)
    pts = np.stack(
        [p[0] + float(r) * np.cos(ang), p[1] + float(r) * np.sin(ang), np.full(ang.size, float(z))],
        axis=1,
    )
    added = 0
    for a, b in zip(pts[:-1], pts[1:]):
        added += _connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, float(width) / 2.0, a, b, rgba)
    return added


def arrow(
    scn: mujoco.MjvScene,
    p_from: Vec,
    p_to: Vec,
    rgba: Vec,
    width: float = 0.03,
) -> int:
    """3-D arrow from ``p_from`` to ``p_to``; 0 geoms for a degenerate (zero-length) arrow."""
    return _connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, float(width) / 2.0, p_from, p_to, rgba)


def ellipse(
    scn: mujoco.MjvScene,
    pos_xy: Vec,
    a: float,
    b: float,
    theta: float,
    rgba: Vec,
    z: float = 0.005,
    thickness: float = 0.004,
) -> int:
    """Flattened ellipsoid footprint with semi-axes ``a``/``b``, rotated ``theta`` about z."""
    p = np.asarray(pos_xy, dtype=float)
    geom = _init_geom(
        scn,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        (float(a), float(b), float(thickness) / 2.0),
        (p[0], p[1], float(z)),
        _rot_z(theta),
        rgba,
    )
    return 0 if geom is None else 1


def path(
    scn: mujoco.MjvScene,
    points_xy: np.ndarray,
    rgba: Vec,
    width: float = 0.015,
    z: float = 0.01,
) -> int:
    """Polyline on the floor through ``points_xy`` ``(N, 2)`` as capsule segments."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0
    added = 0
    for a, b in zip(pts[:-1], pts[1:]):
        added += _connector(
            scn,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(width) / 2.0,
            (a[0], a[1], float(z)),
            (b[0], b[1], float(z)),
            rgba,
        )
    return added


def trail(
    scn: mujoco.MjvScene,
    points_xy: np.ndarray,
    rgba: Vec,
    width: float = 0.012,
    z: float = 0.008,
) -> int:
    """Fading floor trail through ``points_xy``: alpha ramps 0 -> ``rgba[3]`` along the path."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0
    base = np.asarray(rgba, dtype=float)
    n_seg = pts.shape[0] - 1
    added = 0
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        seg_rgba = (base[0], base[1], base[2], base[3] * (i + 1) / n_seg)
        added += _connector(
            scn,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(width) / 2.0,
            (a[0], a[1], float(z)),
            (b[0], b[1], float(z)),
            seg_rgba,
        )
    return added


def pedestrian(
    scn: mujoco.MjvScene,
    p_xy: Vec,
    v_xy: Vec,
    r: float,
    rgba: Vec,
    height: float = 1.65,
    body_scale: float = 0.8,
) -> int:
    """A pedestrian: standing capsule body, heading arrow, and a translucent floor disc.

    The floor disc has radius ``r`` (the pedestrian's own footprint) and the body capsule
    ``body_scale * r``, so a keep-out boundary drawn separately at the inflated radius stays
    visibly wider than the person. The arrow is ``0.5 * |v|`` long, clipped to 0.6 m, and is
    omitted when ``v`` is ~zero.
    """
    p = np.asarray(p_xy, dtype=float)
    v = np.asarray(v_xy, dtype=float).reshape(-1)[:2]
    body_r = float(body_scale) * float(r)
    half_len = max(0.5 * (float(height) - 2.0 * body_r), 1e-3)
    added = 0
    geom = _init_geom(
        scn,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        (body_r, half_len, 0.0),
        (p[0], p[1], float(height) / 2.0),
        np.eye(3),
        rgba,
    )
    if geom is not None:
        added += 1
    speed = float(np.linalg.norm(v))
    if speed > 1e-6:
        length = min(0.5 * speed, 0.6)
        tip = p + v / speed * length
        added += arrow(scn, (p[0], p[1], 0.06), (tip[0], tip[1], 0.06), rgba, width=0.05)
    base = np.asarray(rgba, dtype=float)
    added += disc(scn, p, float(r), (base[0], base[1], base[2], 0.4))
    return added


BEACON_POLE_RADIUS = 0.01
BEACON_POLE_HEIGHT = 1.5
BEACON_POLE_ALPHA = 0.5  # of the sphere's own alpha


def beacon(scn: mujoco.MjvScene, pos_xy: Vec, r: float, rgba: Vec) -> int:
    """Goal beacon: translucent sphere of radius ``r`` plus a hairline column to 1.5 m.

    The sphere marks the spot and the pole makes it findable from across the scene; the
    goal *region* is drawn separately (as floor rings) so the sphere can stay small.
    """
    p = np.asarray(pos_xy, dtype=float)
    added = 0
    geom = _init_geom(
        scn,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        (float(r), 0.0, 0.0),
        (p[0], p[1], float(r)),
        np.eye(3),
        rgba,
    )
    if geom is not None:
        added += 1
    base = np.asarray(rgba, dtype=float)
    column = (base[0], base[1], base[2], BEACON_POLE_ALPHA * float(base[3]))
    added += _connector(
        scn,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        BEACON_POLE_RADIUS,
        (p[0], p[1], 0.0),
        (p[0], p[1], BEACON_POLE_HEIGHT),
        column,
    )
    return added


# --------------------------------------------------------------------------- camera
class CameraSchedule:
    """A never-cutting free camera: lagged look-at, fixed framing, slowly drifting azimuth.

    ``update(k, target_xy)`` returns an ``mjCAMERA_FREE`` camera whose ``lookat`` chases
    ``target_xy`` (at z = 0.8) through a first-order lag of time constant ``lag_s``, which
    filters the gait jitter out of the shot. The azimuth drifts at ``drift_deg_per_s``; when
    a ``heading`` is supplied the camera is additionally biased to sit behind-left of the
    walking direction, blended 50/50 with the drift so it never swings fast.

    ``k`` is the *logged step index* (as the markers callback receives it), so the lag is
    integrated over the actual elapsed ``dt * (k - k_prev)`` even when frames are skipped.
    The same ``MjvCamera`` instance is returned every call -- use it before the next update.

    The azimuth is a filtered state like the look-at, rate-limited to
    :data:`MAX_AZIMUTH_RATE_DEG_S`. Without that limit a heading that flips sign (a robot
    sidestepping, or one whose velocity passes through zero) moves the *target* azimuth by
    up to 180 deg and the shot whips around in a single frame.
    """

    LOOKAT_Z = 0.8
    BEHIND_LEFT_DEG = 180.0 - 35.0
    MAX_AZIMUTH_RATE_DEG_S = 30.0

    def __init__(
        self,
        distance: float,
        elevation: float,
        azimuth0: float = 135.0,
        drift_deg_per_s: float = 0.6,
        lag_s: float = 0.4,
        dt: float = 0.02,
    ) -> None:
        self.distance = float(distance)
        self.elevation = float(elevation)
        self.azimuth0 = float(azimuth0)
        self.drift_deg_per_s = float(drift_deg_per_s)
        self.lag_s = float(lag_s)
        self.dt = float(dt)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.trackbodyid = -1
        self._lookat = np.array([0.0, 0.0, self.LOOKAT_Z])
        self._azimuth = self.azimuth0
        self._last_k: Optional[int] = None

    @staticmethod
    def _wrap180(deg: float) -> float:
        return float((deg + 180.0) % 360.0 - 180.0)

    def update(self, k: int, target_xy: Vec, heading: Optional[float] = None) -> mujoco.MjvCamera:
        """Camera for logged step ``k`` looking at ``target_xy`` (optionally heading-biased)."""
        target = np.asarray(target_xy, dtype=float).reshape(-1)[:2]
        goal = np.array([target[0], target[1], self.LOOKAT_Z])
        last_k = self._last_k
        first = last_k is None or k <= last_k
        elapsed = 0.0 if last_k is None else self.dt * (k - last_k)
        if first:
            self._lookat = goal  # first frame (or a reset): snap, do not ease in from the origin
        else:
            alpha = 1.0 - float(np.exp(-elapsed / max(self.lag_s, 1e-6)))
            self._lookat = self._lookat + alpha * (goal - self._lookat)

        azimuth = self.azimuth0 + self.drift_deg_per_s * self.dt * float(k)
        if heading is not None:
            behind_left = float(np.degrees(float(heading))) + self.BEHIND_LEFT_DEG
            azimuth = azimuth + 0.5 * self._wrap180(behind_left - azimuth)
        if first:
            self._azimuth = self._wrap180(azimuth)
        else:
            step = self._wrap180(azimuth - self._azimuth)
            limit = self.MAX_AZIMUTH_RATE_DEG_S * elapsed
            self._azimuth = self._wrap180(self._azimuth + max(-limit, min(limit, step)))
        self._last_k = int(k)

        self._cam.lookat[:] = self._lookat
        self._cam.distance = self.distance
        self._cam.elevation = self.elevation
        self._cam.azimuth = self._azimuth
        return self._cam


# --------------------------------------------------------------------------- HUD
# The band is rgb(24, 26, 30) at alpha 0.85 over mid-gray, pre-composited because the HUD
# is concatenated below the 3-D view rather than blended into it.
_BAND_RGB = tuple(0.85 * np.array([24.0, 26.0, 30.0]) / 255.0 + 0.15 * 0.5)
_FONT = "DejaVu Sans"
_TEXT = "#e8eaee"
_DIM = "#9aa0a8"
_GRID = "#4a4f57"
_PILL_BG = "#2d3138"
_HUD_WINDOW_S = 6.0
_INTERVENTION_FULL_SCALE = 0.5  # m/s at the right end of the bar
_MAX_PILLS = 5
_PILL_X0, _PILL_X1 = 0.705, 0.995  # the right block's horizontal extent
_PILL_Y_SINGLE = 0.17  # one row of pills sits at the bottom of the band
_PILL_Y_ROWS = (0.40, 0.13)  # two rows: below the intervention bar
_MIN_PILL_SCALE = 0.5  # pills shrink to this fraction of the base size before giving up
_HUD_CACHE_MAX = 2  # figures are expensive; keep the working set, close the rest


class HudRenderer:
    """Cached matplotlib HUD strip: figure and artists are built once and updated per frame.

    ``draw`` returns an ``(height, width, 3)`` uint8 array. Layout: a left text block
    (clock, mode, ``h_min``), a centre 6 s scrolling ``h_min`` plot with the zero line and
    a playhead, and a right block with the intervention bar and status pills.
    """

    def __init__(self, width: int, height: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle

        self.width = int(width)
        self.height = int(height)
        scale = self.width / 1280.0
        self._big = 22.0 * scale  # >= 30 px em at width 1280
        self._mid = 16.0 * scale  # >= 22 px em at width 1280
        self._small = 13.0 * scale

        self._fig = plt.figure(figsize=(self.width / 100.0, self.height / 100.0), dpi=100)
        self._fig.patch.set_facecolor(_BAND_RGB)

        self._t_text = self._fig.text(
            0.016, 0.70, "", color=_TEXT, fontsize=self._big, va="center", fontname=_FONT
        )
        self._mode_text = self._fig.text(
            0.016, 0.44, "", color=_DIM, fontsize=self._mid, va="center", fontname=_FONT
        )
        self._h_text = self._fig.text(
            0.016, 0.17, "", color=_TEXT, fontsize=self._big, va="center", fontname=_FONT
        )

        self._ax = self._fig.add_axes((0.27, 0.20, 0.40, 0.60))
        self._ax.set_facecolor("#1b1e23")
        for side in ("top", "right"):
            self._ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            self._ax.spines[side].set_color(_GRID)
        self._ax.tick_params(colors=_DIM, labelsize=self._small, length=3, pad=2)
        self._ax.set_xticks([])
        self._zero = self._ax.axhline(0.0, color="#d8dde4", lw=1.2, ls="--", alpha=0.8)
        # The second trace is the comparison run (unfiltered) in the side-by-side renders;
        # it stays empty, and out of the y-limits, unless ``draw`` is given ``h_hist2``.
        (self._trace2,) = self._ax.plot([], [], color="#ff6b6b", lw=1.6, ls="--", alpha=0.95)
        (self._trace,) = self._ax.plot([], [], color="#7fd4ff", lw=2.0)
        (self._playhead,) = self._ax.plot([], [], color=_TEXT, lw=1.4, alpha=0.9)
        (self._dot,) = self._ax.plot([], [], "o", ms=6.0, color=_TEXT)
        # A title rather than an in-axes label: the trace would otherwise run through it.
        self._title = f"h_min, last {int(_HUD_WINDOW_S)} s"
        self._ax.set_title(
            self._title,
            color=_DIM,
            fontsize=self._small,
            loc="left",
            pad=3.0,
            fontname=_FONT,
        )
        self._fill: Optional[Any] = None

        self._bar_caption = self._fig.text(
            _PILL_X0, 0.85, "", color=_DIM, fontsize=self._mid, va="center", fontname=_FONT
        )
        self._bar_ax = self._fig.add_axes((_PILL_X0, 0.58, 0.28, 0.15))
        self._bar_ax.set_xlim(0.0, _INTERVENTION_FULL_SCALE)
        self._bar_ax.set_ylim(0.0, 1.0)
        self._bar_ax.axis("off")
        self._bar_full_scale = _INTERVENTION_FULL_SCALE
        self._bar_bg = Rectangle((0.0, 0.0), _INTERVENTION_FULL_SCALE, 1.0, facecolor=_PILL_BG)
        self._bar_ax.add_patch(self._bar_bg)
        self._bar = Rectangle((0.0, 0.0), 0.0, 1.0, facecolor="#7fd4ff")
        self._bar_ax.add_patch(self._bar)

        self._pills = [
            self._fig.text(
                _PILL_X0,
                _PILL_Y_SINGLE,
                "",
                color=_TEXT,
                fontsize=self._mid,
                va="center",
                ha="left",
                bbox={"boxstyle": "round,pad=0.32", "facecolor": _PILL_BG, "edgecolor": "none"},
                fontname=_FONT,
            )
            for _ in range(_MAX_PILLS)
        ]
        for pill in self._pills:
            pill.set_visible(False)

    # -- layout helpers ----------------------------------------------------
    def _pill_width_frac(self, text: str, size: float) -> float:
        """Estimated pill width as a fraction of the figure (DejaVu Sans caps, + padding)."""
        em_px = size * 100.0 / 72.0
        return (0.62 * em_px * len(text) + 0.9 * em_px) / self.width

    def _pack_pills(self, entries: List[Tuple[str, str]], size: float) -> List[List[float]]:
        """Greedy row packing: x offsets per row for ``entries`` at font ``size``."""
        rows: List[List[float]] = [[]]
        x = _PILL_X0
        for text, _ in entries:
            w = self._pill_width_frac(text, size)
            if x + w > _PILL_X1 and rows[-1]:
                rows.append([])
                x = _PILL_X0
            rows[-1].append(x)
            x += w + 0.008
        return rows

    def _set_pills(self, entries: List[Tuple[str, str]]) -> None:
        """Lay the pills out in one or two rows, shrinking the font until they all fit.

        A pill is a status flag -- "QP not converged", a slack magnitude -- so dropping one
        silently would hide exactly the frames a viewer is looking for. This shrinks instead,
        and raises if even the smallest size cannot hold them.
        """
        if len(entries) > _MAX_PILLS:
            raise ValueError(
                f"{len(entries)} HUD pills requested but only {_MAX_PILLS} slots exist: "
                f"{[text for text, _ in entries]}"
            )
        size = self._mid
        rows = self._pack_pills(entries, size)
        while len(rows) > 2 and size > _MIN_PILL_SCALE * self._mid:
            size *= 0.9
            rows = self._pack_pills(entries, size)
        if len(rows) > 2:
            raise ValueError(
                f"HUD pills {[text for text, _ in entries]} do not fit in two rows even at "
                f"{_MIN_PILL_SCALE:.0%} of the base font size"
            )
        y_rows = [_PILL_Y_SINGLE] if len(rows) < 2 else _PILL_Y_ROWS
        offsets = [(x, y_rows[r]) for r, row in enumerate(rows) for x in row]
        for pill, (text, colour), (x, y) in zip(self._pills, entries, offsets):
            pill.set_position((x, y))
            pill.set_text(text)
            pill.set_fontsize(size)
            pill.set_color("#14161a")
            bbox = pill.get_bbox_patch()
            if bbox is not None:
                bbox.set_facecolor(colour)
            pill.set_visible(True)
        for pill in self._pills[len(offsets) :]:
            pill.set_visible(False)

    # -- drawing -----------------------------------------------------------
    def draw(
        self,
        t: float,
        h_min: float,
        h_hist: Optional[np.ndarray],
        t_hist: Optional[np.ndarray],
        intervention: float,
        active: bool,
        flags: Dict[str, bool],
        slack: Optional[float] = None,
        mode: str = "CBF-QP",
        h_hist2: Optional[np.ndarray] = None,
        intervention_caption: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        h_label: str = "h_min",
        full_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Render one HUD frame; see :func:`hud_strip` for the argument meanings."""
        colour = h_rgba(h_min, 1.0)[:3]
        self._t_text.set_text(f"t = {float(t):5.1f} s")
        self._mode_text.set_text(str(mode))
        self._h_text.set_text(f"h min = {float(h_min):+.2f}")
        self._h_text.set_color(_rgb(colour))

        if self._fill is not None:
            self._fill.remove()
            self._fill = None
        t_lo, t_hi = float(t) - _HUD_WINDOW_S, float(t)
        if h_hist is not None and t_hist is not None and len(np.asarray(t_hist)) > 0:
            t_all = np.asarray(t_hist, dtype=float).reshape(-1)
            h_all = np.asarray(h_hist, dtype=float).reshape(-1)
            keep = (t_all >= t_lo) & (t_all <= t_hi + 1e-9)
            th, hh = t_all[keep], h_all[keep]
            self._trace.set_data(th, hh)
            if th.size:
                below = [bool(v) for v in hh < 0.0]
                self._fill = self._ax.fill_between(
                    th, hh, 0.0, where=below, color="#d94b56", alpha=0.35, interpolate=True
                )
                lo = min(float(hh.min()) - 0.05, -0.1)
                hi = max(float(hh.max()) + 0.05, 0.4)
            else:
                lo, hi = -0.1, 0.4
        else:
            self._trace.set_data([], [])
            lo, hi = -0.1, 0.4

        # The comparison trace shares ``t_hist``: both runs are logged on the same clock, so a
        # shorter ``h_hist2`` (a comparison run that ended earlier) is simply truncated to fit.
        if h_hist2 is not None and t_hist is not None:
            h_all2 = np.asarray(h_hist2, dtype=float).reshape(-1)
            t_all2 = np.asarray(t_hist, dtype=float).reshape(-1)
            n2 = min(h_all2.size, t_all2.size)
            t2, h2 = t_all2[:n2], h_all2[:n2]
            keep2 = (t2 >= t_lo) & (t2 <= t_hi + 1e-9)
            self._trace2.set_data(t2[keep2], h2[keep2])
            if keep2.any():
                lo = min(lo, float(h2[keep2].min()) - 0.05)
                hi = max(hi, float(h2[keep2].max()) + 0.05)
        else:
            self._trace2.set_data([], [])
        title = f"{h_label}, last {int(_HUD_WINDOW_S)} s" + (
            "   (red dashed: no filter)" if h_hist2 is not None else ""
        )
        if title != self._title:
            self._title = title
            self._ax.set_title(
                title, color=_DIM, fontsize=self._small, loc="left", pad=3.0, fontname=_FONT
            )

        # A caller that knows the whole clip pins the range once: autoscaling per frame makes
        # the trace jump every time the 6 s window slides past a peak.
        if ylim is not None:
            lo, hi = float(ylim[0]), float(ylim[1])
        self._ax.set_xlim(t_lo, t_hi)
        self._ax.set_ylim(lo, hi)
        # the lower tick is dropped when it would crowd the zero line's label
        ticks = [0.0, round(hi, 2)]
        if (0.0 - lo) / max(hi - lo, 1e-9) > 0.18:
            ticks.insert(0, round(lo, 2))
        self._ax.set_yticks(ticks)
        self._playhead.set_data([t_hi, t_hi], [lo, hi])
        self._dot.set_data([t_hi], [float(h_min)])
        self._dot.set_color(_rgb(colour))

        iv = float(intervention)
        self._bar_caption.set_text(
            f"intervention  |dv| = {iv:.2f} m/s"
            if intervention_caption is None
            else str(intervention_caption)
        )
        scale = _INTERVENTION_FULL_SCALE if full_scale is None else max(float(full_scale), 1e-9)
        if scale != self._bar_full_scale:
            self._bar_full_scale = scale
            self._bar_ax.set_xlim(0.0, scale)
            self._bar_bg.set_width(scale)
        self._bar.set_width(min(max(iv, 0.0), scale))
        self._bar.set_facecolor("#7fd4ff" if active else _GRID)

        entries: List[Tuple[str, str]] = []
        if active:
            entries.append(("CBF ACTIVE", "#3fc46a"))
        for name, on in (flags or {}).items():
            if on:
                entries.append((str(name).upper(), "#f0a830"))
        if slack is not None:
            hot = float(slack) > 1e-3
            entries.append((f"SLACK {float(slack):.3f}", "#d94b56" if hot else _DIM))
        self._set_pills(entries)

        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
        return np.ascontiguousarray(buf[:, :, :3]).copy()

    def close(self) -> None:
        from matplotlib import pyplot as plt

        plt.close(self._fig)


_HUD_CACHE: "OrderedDict[Tuple[int, int], HudRenderer]" = OrderedDict()


def hud_strip(
    width: int,
    height: int,
    t: float,
    h_min: float,
    h_hist: Optional[np.ndarray],
    t_hist: Optional[np.ndarray],
    intervention: float,
    active: bool,
    flags: Dict[str, bool],
    slack: Optional[float] = None,
    mode: str = "CBF-QP",
    h_hist2: Optional[np.ndarray] = None,
    intervention_caption: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    h_label: str = "h_min",
    full_scale: Optional[float] = None,
) -> np.ndarray:
    """One HUD strip as an ``(height, width, 3)`` uint8 array.

    ``t`` is the current time and ``h_min`` the current minimum barrier value (it colours
    the numeric readout and the plot's playhead dot). ``h_hist``/``t_hist`` are the
    history arrays, windowed here to the last 6 s. ``intervention`` is
    ``|v_safe - v_nom|`` in m/s, drawn as a bar of full scale ``full_scale`` (0.5 by default;
    pass the control bound when the filtered variable is an acceleration rather than a
    velocity, or the bar pegs through every braking manoeuvre). ``active`` lights the "CBF ACTIVE"
    pill, ``flags`` adds one amber pill per true entry (short upper-case keys fit best,
    e.g. ``{"mppi fallback": False, "qp fail": True}``), and ``slack`` adds a slack pill
    for the relaxed-QP runs. ``mode`` is the small label under the clock.

    ``h_hist2`` overlays a second ``h_min`` history (the unfiltered comparison run) as a red
    dashed trace on the same clock as ``t_hist``, and ``intervention_caption`` replaces the
    bar's caption for runs where the filtered variable is not a velocity. ``ylim`` pins the
    plot's vertical range instead of autoscaling it to the visible 6 s, and ``h_label`` names
    the quantity plotted (for a caller that plots a reparametrised barrier).

    Figures are cached per ``(width, height)`` because this runs per frame; the cache holds
    the most recent few and closes the rest.
    """
    key = (int(width), int(height))
    renderer = _HUD_CACHE.pop(key, None)
    if renderer is None:
        renderer = HudRenderer(*key)
    _HUD_CACHE[key] = renderer  # re-inserted last: the dict doubles as an LRU order
    while len(_HUD_CACHE) > _HUD_CACHE_MAX:
        _, evicted = _HUD_CACHE.popitem(last=False)
        evicted.close()
    return renderer.draw(
        t,
        h_min,
        h_hist,
        t_hist,
        intervention,
        active,
        flags,
        slack,
        mode,
        h_hist2,
        intervention_caption,
        ylim,
        h_label,
    )


# --------------------------------------------------------------------------- compositing
def _dejavu_candidates(name: str) -> List[Path]:
    """Where a DejaVu ``.ttf`` might live: matplotlib's bundled copy first, then the system.

    ``find_spec`` locates matplotlib's data directory *without importing* matplotlib, so
    this module keeps matplotlib out of its import path (see the module docstring).
    """
    candidates: List[Path] = []
    try:
        spec = find_spec("matplotlib")
    except (ImportError, ValueError):  # pragma: no cover - defensive
        spec = None
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf" / name)
    return candidates + [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
        Path("/opt/homebrew/share/fonts") / name,
        Path("/Library/Fonts") / name,
    ]


def _label_font(size: int):
    """DejaVu Sans Bold at ``size`` px for PIL, falling back to PIL's built-in bitmap font."""
    from PIL import ImageFont

    for cand in _dejavu_candidates("DejaVuSans-Bold.ttf"):
        if cand.is_file():
            return ImageFont.truetype(str(cand), size)
    return ImageFont.load_default()


def compose_panels(
    panels: List[np.ndarray],
    labels: List[str],
    hud: Optional[np.ndarray],
    gap: int = 8,
    bg: Tuple[int, int, int] = (18, 18, 20),
    captions: Optional[Sequence[Optional[str]]] = None,
) -> np.ndarray:
    """Lay panels out side by side with a label pill each, and the HUD strip underneath.

    Panels are concatenated horizontally with ``gap`` pixels of ``bg`` between them; the
    HUD (any width) goes below, and both rows are centred on the widest of the two.
    ``captions`` optionally adds a second, smaller pill under a panel's label (``None``
    for a panel that has none) -- for example to say that one panel is frozen on its last
    frame while the other runs on. Returns an ``(H, W, 3)`` uint8 array.
    """
    from PIL import Image, ImageDraw

    if not panels:
        raise ValueError("compose_panels needs at least one panel")
    if len(labels) != len(panels):
        raise ValueError(f"got {len(panels)} panels but {len(labels)} labels")
    frames = [np.asarray(p, dtype=np.uint8) for p in panels]
    row_w = sum(f.shape[1] for f in frames) + gap * (len(frames) - 1)
    row_h = max(f.shape[0] for f in frames)
    hud_arr = None if hud is None else np.asarray(hud, dtype=np.uint8)
    hud_w = 0 if hud_arr is None else hud_arr.shape[1]
    hud_h = 0 if hud_arr is None else hud_arr.shape[0]
    out_w = max(row_w, hud_w)
    out_h = row_h + (gap + hud_h if hud_arr is not None else 0)

    canvas = Image.new("RGB", (out_w, out_h), tuple(int(c) for c in bg))
    x = (out_w - row_w) // 2
    boxes = []
    for frame in frames:
        canvas.paste(Image.fromarray(frame), (x, 0))
        boxes.append((x, frame.shape[1]))
        x += frame.shape[1] + gap
    if hud_arr is not None:
        canvas.paste(Image.fromarray(hud_arr), ((out_w - hud_w) // 2, row_h + gap))

    size = max(14, int(round(row_h / 26.0)))
    font = _label_font(size)
    small_size = max(11, int(round(size * 0.72)))
    small_font = _label_font(small_size)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad = max(6, size // 2)
    caption_list = list(captions) if captions is not None else [None] * len(boxes)
    if len(caption_list) != len(boxes):
        raise ValueError(f"got {len(boxes)} panels but {len(caption_list)} captions")

    def pill(x0: int, y0: int, text: str, use_font, fill) -> int:
        """Draw one rounded label at ``(x0, y0)``; returns the bottom edge."""
        try:
            left, top, right, bottom = draw.textbbox((x0, y0), text, font=use_font)
        except AttributeError:  # pragma: no cover - very old Pillow
            right, bottom = x0 + size * len(text) // 2, y0 + size
            left, top = x0, y0
        draw.rounded_rectangle(
            (left - pad, top - pad // 2, right + pad, bottom + pad // 2),
            radius=pad,
            fill=(18, 20, 24, 200),
        )
        draw.text((x0, y0), text, font=use_font, fill=fill)
        return int(bottom)

    for (px, _), text, caption in zip(boxes, labels, caption_list):
        x0, y0 = px + pad + 2, pad + 2
        bottom = pill(x0, y0, text, font, (232, 234, 238, 255))
        if caption:
            pill(x0, bottom + pad + pad // 2, caption, small_font, (200, 205, 212, 255))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(canvas, dtype=np.uint8)


# --------------------------------------------------------------------------- writers
def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: ``imageio_ffmpeg``'s bundled one, else one on ``PATH``."""
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 - any import/lookup failure falls back to PATH
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise RuntimeError(
            "no ffmpeg available: install imageio-ffmpeg (pip install imageio-ffmpeg) "
            "or put ffmpeg on PATH"
        ) from None


class FrameWriter:
    """Stream RGB frames to an MP4 or a palette-optimised GIF via ffmpeg.

    ``fps`` is the rate the added frames represent, so for a 0.02 s simulation rendered
    every other step you pass ``fps=25`` for a real-time MP4. ``speed`` only applies to
    GIFs: the frames are time-compressed by ``speed`` and the GIF is written at
    ``fps * speed``, i.e. ``FrameWriter(p, 10, kind="gif", speed=2.0)`` gives a 20 fps GIF
    playing 2x faster than real time at 10 frames per simulated second. ``gif_fps`` overrides
    that output rate on its own (the playback speed is unchanged), so the same added frames
    can be resampled to a smaller file.

    Frames must all be the same size; an odd width or height is padded by one pixel
    because ``yuv420p`` needs even dimensions.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        fps: float,
        *,
        kind: str = "mp4",
        speed: float = 1.0,
        gif_width: int = 480,
        gif_colors: int = 64,
        gif_fps: Optional[float] = None,
    ) -> None:
        if kind not in ("mp4", "gif"):
            raise ValueError(f"kind must be 'mp4' or 'gif', got {kind!r}")
        self.path = Path(path)
        self.fps = float(fps)
        self.kind = kind
        self.speed = float(speed)
        self.gif_width = int(gif_width)
        self.gif_colors = int(gif_colors)
        self.gif_fps = None if gif_fps is None else float(gif_fps)
        self.n_frames = 0
        self._size: Optional[Tuple[int, int]] = None
        self._proc: Optional[subprocess.Popen] = None
        self._tmp: Optional[Path] = None
        self._closed = False

    # -- ffmpeg plumbing ---------------------------------------------------
    def _target(self) -> Path:
        if self.kind == "mp4":
            return self.path
        if self._tmp is None:
            fd, name = tempfile.mkstemp(prefix="cbfkit_showcase_", suffix=".mp4")
            os.close(fd)
            self._tmp = Path(name)
        return self._tmp

    def _start(self, width: int, height: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        crf = "18" if self.kind == "mp4" else "12"  # the GIF intermediate stays near-lossless
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            f"{self.fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{self.fps:g}",
            str(self._target()),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _gif_pass(self, src: Path) -> None:
        # The time-compressed stream runs at ``fps * speed``; ``gif_fps`` resamples that to a
        # different output rate (dropping or duplicating frames) without changing the speed.
        out_fps = self.fps * self.speed if self.gif_fps is None else self.gif_fps
        vf = (
            f"setpts=(PTS-STARTPTS)/{self.speed:g},fps={out_fps:g},"
            f"scale={self.gif_width}:-1:flags=lanczos,split[a][b];"
            f"[a]palettegen=max_colors={self.gif_colors}:stats_mode=diff[p];"
            f"[b][p]paletteuse=dither=none:diff_mode=rectangle"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-vf", vf, str(self.path)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode(errors="replace")[:2000]
            raise RuntimeError(
                f"ffmpeg's GIF palette pass failed ({exc.returncode}) for {self.path}: {detail}"
            ) from exc

    # -- public API --------------------------------------------------------
    def add(self, frame_rgb_uint8: np.ndarray) -> None:
        """Append one ``(H, W, 3)`` uint8 RGB frame."""
        if self._closed:
            raise RuntimeError("FrameWriter is closed")
        frame = np.ascontiguousarray(np.asarray(frame_rgb_uint8, dtype=np.uint8))
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB frame, got shape {frame.shape}")
        if frame.shape[0] % 2 or frame.shape[1] % 2:
            pad_h, pad_w = frame.shape[0] % 2, frame.shape[1] % 2
            frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)))
        h, w = frame.shape[:2]
        if self._size is None:
            self._size = (w, h)
            self._start(w, h)
        elif self._size != (w, h):
            raise ValueError(f"frame size changed from {self._size} to {(w, h)}")
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            # ffmpeg died mid-stream; its stderr says why, and without it the traceback is
            # just "broken pipe" on frame N.
            raise RuntimeError(f"ffmpeg exited while writing {self.path}: {self._drain()}") from exc
        self.n_frames += 1

    def _drain(self) -> str:
        """Whatever ffmpeg wrote to stderr, once it has exited."""
        if self._proc is None:
            return ""
        try:
            err = self._proc.stderr.read() if self._proc.stderr is not None else b""
            self._proc.wait(timeout=10)
        except (OSError, ValueError, subprocess.TimeoutExpired):  # pragma: no cover - defensive
            return ""
        return err.decode(errors="replace")[:2000]

    def close(self) -> str:
        """Finish the encode (and the GIF palette pass) and return the output path."""
        if self._closed:
            return str(self.path)
        self._closed = True
        if self._proc is None:
            raise RuntimeError(f"no frames were added, nothing to write to {self.path}")
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg exited while closing {self.path}: {self._drain()}") from exc
        err = self._proc.stderr.read() if self._proc.stderr is not None else b""
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed ({code}): {err.decode(errors='replace')[:2000]}")
        try:
            if self.kind == "gif":
                assert self._tmp is not None
                self._gif_pass(self._tmp)
        finally:
            if self._tmp is not None and self._tmp.exists():
                self._tmp.unlink()
        return str(self.path)

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            if self._proc is not None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except BrokenPipeError:  # ffmpeg already gone
                    pass
                self._proc.wait()
            self._closed = True
            if self._tmp is not None and self._tmp.exists():
                self._tmp.unlink()
            # A half-written file is worse than none: a later run would read it as a result.
            if self.path.exists():
                self.path.unlink()
            return
        self.close()


# --------------------------------------------------------------------------- kinematics
def body_positions(
    mj_model: mujoco.MjModel, states: np.ndarray, nq: int, body_ids: Sequence[int]
) -> np.ndarray:
    """Forward-kinematics body positions per logged step: ``(T, len(body_ids), 3)``.

    Only ``qpos`` is written, so this is pure kinematics -- exactly what the overlays need
    and much cheaper than ``mj_forward``.
    """
    S = np.asarray(states, dtype=float)
    ids = np.asarray(body_ids, dtype=int).reshape(-1)
    data = mujoco.MjData(mj_model)
    out = np.empty((S.shape[0], ids.size, 3), dtype=float)
    for k in range(S.shape[0]):
        data.qpos[:] = S[k, : int(nq)]
        mujoco.mj_kinematics(mj_model, data)
        out[k] = data.xpos[ids]
    return out


def pelvis_yaw(states: np.ndarray, nq: int) -> np.ndarray:
    """Realised pelvis yaw per logged step from the free-joint quaternion ``qpos[3:7]``."""
    if int(nq) < 7:
        raise ValueError(f"nq = {nq} has no free-joint quaternion at qpos[3:7]")
    q = np.asarray(states, dtype=float)[:, : int(nq)]
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
