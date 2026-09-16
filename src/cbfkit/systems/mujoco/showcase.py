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


from ._showcase_camera import (
    CameraSchedule,
)
from ._showcase_geometry import (
    _AMBER,
    _GREEN,
    _RED,
    BEACON_POLE_ALPHA,
    BEACON_POLE_HEIGHT,
    BEACON_POLE_RADIUS,
    _connector,
    _init_geom,
    _rgb,
    _rot_z,
    arrow,
    beacon,
    disc,
    ellipse,
    h_rgba,
    path,
    pedestrian,
    ring,
    trail,
)
from ._showcase_hud import (
    _BAND_RGB,
    _DIM,
    _FONT,
    _GRID,
    _HUD_CACHE,
    _HUD_CACHE_MAX,
    _HUD_WINDOW_S,
    _INTERVENTION_FULL_SCALE,
    _MAX_PILLS,
    _MIN_PILL_SCALE,
    _PILL_BG,
    _PILL_X0,
    _PILL_X1,
    _PILL_Y_ROWS,
    _PILL_Y_SINGLE,
    _TEXT,
    HudRenderer,
    hud_strip,
)
from ._showcase_output import (
    FrameWriter,
    _dejavu_candidates,
    _label_font,
    compose_panels,
    ffmpeg_exe,
)


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
