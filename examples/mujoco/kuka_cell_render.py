"""Presentation renders of the KUKA cell logs written by ``kuka_cell.py simulate``.

    python examples/mujoco/kuka_cell_render.py --npz results/showcase/kuka_cell.npz
           [--unfiltered-npz results/showcase/kuka_cell_unfiltered.npz --side-by-side]
           [--out DIR] [--stills] [--max-seconds S] [--no-mp4] [--no-gif]
           [--gif-width N] [--gif-colors N] [--gif-fps F] [--gif-no-hud] [--no-drift]
           [--floor grid|checker|plain]

Reuses the G1 showcase machinery (:mod:`cbfkit.systems.mujoco.showcase` and the
:class:`~examples.mujoco.g1_showcase_render.Run` loader, strides and encoders) with a
fixed three-quarter camera over the cell and the arm-specific overlays:

* the worker's five capsules drawn again at the keep-out radius ``r_j + d_0`` and coloured
  by the smallest barrier value that capsule is part of (green -> amber -> red);
* the arm sphere of the closest (sphere, capsule) pair and a connector to its capsule --
  the constraint the QP is working on;
* a fading 3-D trail of the tool over the last :data:`TRAIL_S` seconds and a small ghost
  sphere at the nominal reference the arm is trying to reach;
* on a step with contact, a red flash on the arm sphere involved.

The HUD strip shows ``h_min``, the intervention ``max_i |qdot_safe_i - qdot_nom_i|`` in
rad/s and a ``CONTACT`` pill. In a side-by-side the unfiltered run is the left panel
("no filter") and the filtered run the right ("+ CBFKit safety filter"), the HUD carrying
the filtered run's readouts with the unfiltered ``h_min`` as the second trace.
"""

import argparse
import contextlib
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from cbfkit.systems.mujoco import kuka_cell as kc  # noqa: E402
from cbfkit.systems.mujoco import showcase  # noqa: E402
from cbfkit.systems.mujoco._showcase_geometry import _connector, _init_geom, h_rgba  # noqa: E402
from examples.mujoco.g1_showcase_render import (  # noqa: E402
    GIF_COLORS,
    GIF_FPS_OUT,
    GIF_WIDTH,
    HUD_H,
    MAX_GEOM,
    PANEL_H,
    PANEL_W,
    SBS_HUD_H,
    SBS_HUD_W,
    SBS_PANEL_H,
    SBS_PANEL_W,
    STILL_FRACTIONS,
    Run,
    _encode,
    _live_length,
    _strides,
    _window_indices,
    _write_stills,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SHOWCASE_DIR = os.path.join(RESULTS_DIR, "showcase")
EXAMPLE = "kuka_cell"
LABEL_CBF = "+ CBFKit safety filter"
LABEL_UNFILTERED = "no filter"
MODE_CBF = "CBF-QP on joint velocities"
MODE_UNFILTERED = "no filter (nominal)"

# Camera: fixed look-at over the cell, slow azimuth drift so the shot is not a still.
LOOKAT = (0.40, 0.05, 0.72)
CAM_DISTANCE = 3.0
CAM_ELEVATION = -18.0
CAM_AZIMUTH0 = 60.0  # camera on the bin side; the worker crosses behind the arm
DRIFT_DEG_PER_S = 0.5

TRAIL_S = 2.0
TRAIL_POINTS = 40
TRAIL_R = 0.008
KEEPOUT_ALPHA = 0.22
PAIR_ALPHA = 0.75
CONNECTOR_R = 0.006
GHOST_R = 0.03
GHOST_RGBA = (0.25, 0.45, 0.95, 0.55)
TRAIL_RGBA = (0.20, 0.55, 0.95, 0.85)
CONTACT_RGBA = (0.90, 0.10, 0.10, 0.55)


# --------------------------------------------------------------------------- per-run prep
class _Geometry:
    """Kinematics on the *physics* model, so the arm sphere ids match the simulation's."""

    def __init__(self, offline: bool, arm_geoms: Optional[np.ndarray] = None) -> None:
        self.model = kc.load_cell(offline=offline)
        self.data = mujoco.MjData(self.model)
        if arm_geoms is None:  # a log that predates the ``arm_geoms`` key: the current set
            plant_geoms = []
            for body, k in kc.ARM_SPHERE_GEOMS:
                geoms = kc._collision_geoms(self.model, body)
                plant_geoms.append(int(geoms[k]))
            arm_geoms = np.asarray(plant_geoms)
        self.arm_geoms = np.asarray(arm_geoms, dtype=int).reshape(-1)
        self.arm_r = np.asarray(self.model.geom_size[self.arm_geoms, 0], dtype=float)

    def spheres(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        return np.array(self.data.geom_xpos[self.arm_geoms])


def _worker_pose(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """``(mocap_pos, mocap_quat, yaw)`` for one logged worker row ``(px, py, vx, vy)``."""
    yaw = float(np.asarray(kc.worker_yaw(w[2:])))
    quat = np.asarray(kc.worker_quat(yaw), dtype=float)
    return np.array([w[0], w[1], 0.0]), quat, yaw


def _capsules_world(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _pos, _quat, yaw = _worker_pose(w)
    a, b, r = kc.worker_capsules(np.asarray(w[:2], dtype=float), yaw)
    return np.asarray(a), np.asarray(b), np.asarray(r)


def _closest_on_segment(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = max(float(ab @ ab), 1e-12)
    s = min(max(float((c - a) @ ab) / denom, 0.0), 1.0)
    return a + s * ab


def _prepare(run: Run, offline: bool) -> Dict[str, Any]:
    """Everything an overlay needs that is not in the npz: reference path, worker poses."""
    import jax

    worker = np.asarray(run.get("worker"), dtype=float).reshape(run.T, 4)
    tool = np.asarray(run.get("tool"), dtype=float).reshape(run.T, 3)
    contact = run.get("contact")
    contact = np.zeros(run.T, dtype=bool) if contact is None else np.asarray(contact, dtype=bool)
    ref = jax.jit(kc.pick_place_reference)
    p_ref = np.stack([np.asarray(ref(t)[0]) for t in run.t])
    qdot_nom = run.get("qdot_nom")
    qdot_safe = run.get("qdot_safe")
    if qdot_nom is not None and qdot_safe is not None:
        intervention = np.abs(np.asarray(qdot_safe) - np.asarray(qdot_nom)).max(axis=1)
    else:
        intervention = np.zeros(run.T)
    d0 = run.scalar("d0", kc.D0)
    return dict(
        worker=worker,
        tool=tool,
        contact=contact,
        p_ref=p_ref,
        intervention=intervention,
        d0=d0,
        geometry=_Geometry(offline, run.get("arm_geoms")),
        n_spheres=run.h.shape[1] // kc.N_CAPSULES,  # the log's sphere set, not the module's
        trail_n=max(2, int(round(TRAIL_S / run.dt))),
    )


# --------------------------------------------------------------------------- overlays
def _sphere(scn: mujoco.MjvScene, centre: np.ndarray, r: float, rgba) -> int:
    g = _init_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (r, r, r), centre, np.eye(3), rgba)
    return 0 if g is None else 1


def _keepout_capsules(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    a, b, r = _capsules_world(prep["worker"][k])
    h = run.h[k].reshape(prep["n_spheres"], kc.N_CAPSULES)
    for j in range(kc.N_CAPSULES):
        rgba = h_rgba(float(h[:, j].min()), KEEPOUT_ALPHA)
        radius = float(r[j] + prep["d0"])
        if np.linalg.norm(b[j] - a[j]) < 1e-9:
            _sphere(scn, a[j], radius, rgba)
        else:
            _connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a[j], b[j], rgba)


def _active_pair(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    geo: _Geometry = prep["geometry"]
    centres = geo.spheres(run.states[k, : run.nq])
    h = run.h[k]
    idx = int(np.argmin(h))
    i, j = divmod(idx, kc.N_CAPSULES)
    a, b, _r = _capsules_world(prep["worker"][k])
    rgba = h_rgba(float(h[idx]), PAIR_ALPHA)
    _sphere(scn, centres[i], float(geo.arm_r[i]) + 0.004, rgba)
    target = _closest_on_segment(centres[i], a[j], b[j])
    _connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, CONNECTOR_R, centres[i], target, rgba)
    if prep["contact"][k]:
        _sphere(scn, centres[i], float(geo.arm_r[i]) + 0.05, CONTACT_RGBA)


def _tool_trail(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    lo = max(0, k - prep["trail_n"])
    pts = prep["tool"][lo : k + 1]
    if len(pts) > TRAIL_POINTS:
        sel = np.linspace(0, len(pts) - 1, TRAIL_POINTS).round().astype(int)
        pts = pts[sel]
    n_seg = max(len(pts) - 1, 1)
    for s, (p, q) in enumerate(zip(pts[:-1], pts[1:])):
        rgba = (*TRAIL_RGBA[:3], TRAIL_RGBA[3] * (s + 1) / n_seg)
        _connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, TRAIL_R, p, q, rgba)
    _sphere(scn, prep["p_ref"][k], GHOST_R, GHOST_RGBA)


def overlay(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    _keepout_capsules(scn, run, k, prep)
    _tool_trail(scn, run, k, prep)
    _active_pair(scn, run, k, prep)


# --------------------------------------------------------------------------- panels
class _Panel:
    """One 3-D view of one run: render model, data, renderer, overlay, shared camera path."""

    def __init__(
        self,
        run: Run,
        width: int,
        height: int,
        offline: bool,
        floor: str,
        azimuths: np.ndarray,
        k0: int,
    ) -> None:
        self.run = run
        self.model = kc.build_spec(offline=offline, showcase_floor=floor).compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width, max_geom=MAX_GEOM)
        self.prep = _prepare(run, offline)
        self.k0 = int(k0)
        self.azimuths = azimuths
        self.freeze_at = min(int(run.n_live), run.T - 1)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.trackbodyid = -1
        self._cam.lookat[:] = LOOKAT
        self._cam.distance = CAM_DISTANCE
        self._cam.elevation = CAM_ELEVATION

    def step(self, k: int) -> int:
        return self.run.clip(min(int(k), self.freeze_at))

    def frozen(self, k: int) -> bool:
        return int(k) > self.freeze_at

    def frame(self, k: int) -> np.ndarray:
        kk = self.step(k)
        self._cam.azimuth = float(
            self.azimuths[min(max(int(k) - self.k0, 0), len(self.azimuths) - 1)]
        )
        nq, nv = self.run.nq, self.run.nv
        self.data.qpos[:] = self.run.states[kk, :nq]
        self.data.qvel[:] = self.run.states[kk, nq : nq + nv]
        pos, quat, _yaw = _worker_pose(self.prep["worker"][kk])
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = quat
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self._cam)
        overlay(self.renderer.scene, self.run, kk, self.prep)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def _end_caption(run: Run) -> str:
    t_end = run.clip(int(run.n_live)) * run.dt
    return f"QP error at {t_end:.1f} s, frozen" if run.n_live < run.T else "frozen"


def _hud_kwargs(run: Run, prep: Dict[str, Any], k: int, mode: str) -> Dict[str, Any]:
    kk = run.clip(k)
    value = float(prep["intervention"][kk])
    status = run.get("solver_status")
    flags: Dict[str, bool] = {"CONTACT": bool(prep["contact"][kk])}
    if status is not None:
        flags["QP not converged"] = int(np.asarray(status).reshape(-1)[kk]) != 1
    # A relaxed run logs one slack column per barrier after the n_u velocity columns.
    slack: Optional[float] = None
    sol = run.get("sol")
    if run.relax and sol is not None and sol.ndim == 2 and sol.shape[1] > run.n_u:
        slack = float(np.max(np.asarray(sol[kk], dtype=float)[run.n_u :]))
    return dict(
        slack=slack,
        t=float(run.t[kk]),
        h_min=float(run.h_min[kk]),
        h_hist=run.h_min[: kk + 1],
        t_hist=run.t[: kk + 1],
        intervention=value,
        active=value > 1e-3,
        flags=flags,
        mode=mode,
        intervention_caption=f"intervention  {value:.2f} rad/s",
        h_label="h_min (m)",
        full_scale=float(kc.QD_MAX),
    )


def _hud_ylim(run: Run, k0: int, k1: int) -> Tuple[float, float]:
    segment = run.h_min[k0:k1]
    if segment.size == 0:
        segment = run.h_min
    lo = min(-0.25, 1.1 * float(segment.min()))
    hi = max(0.6, float(np.percentile(segment, 90.0)))
    return lo, hi


def _composites(
    runs: List[Run],
    labels: List[str],
    ks: Sequence[int],
    *,
    panel_wh: Tuple[int, int],
    hud_wh: Tuple[int, int],
    hud_run: Run,
    compare_run: Optional[Run],
    downscale: Optional[Tuple[int, int]],
    ylim: Tuple[float, float],
    offline: bool,
    floor: str,
    k0: int,
    k1: int,
    need_hud: bool,
    plain_too: bool,
    drift: float,
):
    """Yield ``(k, hud_frame, plain_frame)`` per ``k``; each 3-D frame rendered once."""
    from PIL import Image

    azimuths = CAM_AZIMUTH0 + drift * hud_run.dt * np.arange(max(k1 - k0, 1))
    with contextlib.ExitStack() as stack:
        panels = [
            stack.enter_context(
                contextlib.closing(
                    _Panel(r, panel_wh[0], panel_wh[1], offline, floor, azimuths, k0)
                )
            )
            for r in runs
        ]
        hud = (
            stack.enter_context(contextlib.closing(showcase.HudRenderer(*hud_wh)))
            if need_hud
            else None
        )
        hud_prep = next(p.prep for p in panels if p.run is hud_run)
        mode = MODE_UNFILTERED if hud_run.unfiltered else MODE_CBF
        for k in ks:
            images = [p.frame(k) for p in panels]
            if downscale is not None:
                images = [
                    np.asarray(Image.fromarray(im).resize(downscale, Image.LANCZOS))
                    for im in images
                ]
            captions = [_end_caption(p.run) if p.frozen(k) else None for p in panels]
            if not any(captions):
                captions = None
            plain = (
                showcase.compose_panels(images, labels, None, captions=captions)
                if plain_too or hud is None
                else None
            )
            if hud is not None:
                kwargs = _hud_kwargs(hud_run, hud_prep, k, mode)
                kwargs["ylim"] = ylim
                if compare_run is not None:
                    kk2 = compare_run.clip(k)
                    kwargs["h_hist2"] = compare_run.h_min[: kk2 + 1]
                main = showcase.compose_panels(
                    images, labels, hud.draw(**kwargs), captions=captions
                )
            else:
                assert plain is not None
                main = plain
            yield k, main, (plain if plain_too and plain is not None else main)


# --------------------------------------------------------------------------- public API
def render(
    npz: str,
    *,
    unfiltered_npz: Optional[str] = None,
    out_dir: str = SHOWCASE_DIR,
    side_by_side: bool = False,
    mp4: bool = True,
    gif: bool = True,
    stills: bool = False,
    max_seconds: Optional[float] = None,
    gif_width: int = GIF_WIDTH,
    gif_colors: int = GIF_COLORS,
    gif_fps: float = GIF_FPS_OUT,
    gif_no_hud: bool = False,
    no_drift: bool = False,
    floor: str = "grid",
) -> List[str]:
    """Render ``npz`` (and with ``side_by_side`` its unfiltered twin); return the output paths."""
    offline = bool(os.environ.get("CBFKIT_ASSETS_OFFLINE"))
    cbf = Run(npz)
    if cbf.example and cbf.example != EXAMPLE:
        raise ValueError(f"{npz} was simulated as {cbf.example!r}, not {EXAMPLE!r}")
    unfiltered: Optional[Run] = None
    if unfiltered_npz is not None:
        unfiltered = Run(unfiltered_npz)
        if abs(unfiltered.dt - cbf.dt) > 1e-12:
            raise ValueError(f"dt mismatch: {cbf.dt} vs {unfiltered.dt}")
    if side_by_side and unfiltered is None:
        raise ValueError("side_by_side needs --unfiltered-npz")
    os.makedirs(out_dir, exist_ok=True)
    outputs: List[str] = []
    passes: List[Dict[str, Any]] = [
        dict(
            suffix="",
            runs=[cbf],
            labels=[LABEL_CBF],
            panel_wh=(PANEL_W, PANEL_H),
            hud_wh=(PANEL_W, HUD_H),
            downscale=None,
            compare=None,
            length_runs=[cbf],
        )
    ]
    if side_by_side and unfiltered is not None:
        passes.append(
            dict(
                suffix="_side_by_side",
                runs=[unfiltered, cbf],
                labels=[LABEL_UNFILTERED, LABEL_CBF],
                panel_wh=(PANEL_W, PANEL_H),
                hud_wh=(SBS_HUD_W, SBS_HUD_H),
                downscale=(SBS_PANEL_W, SBS_PANEL_H),
                compare=unfiltered,
                length_runs=[cbf, unfiltered],
            )
        )
    for spec in passes:
        n = _live_length(spec["length_runs"])
        k0, k1, _windowed = _window_indices(n, cbf.dt, None, max_seconds)
        out_base = os.path.join(out_dir, f"{EXAMPLE}{spec['suffix']}")
        if stills:
            hi = min(k1, max(k0 + 1, int(cbf.n_live)))
            still_ks = [k0 + int(round(f * (hi - 1 - k0))) for f in STILL_FRACTIONS]
            ks = sorted(set(still_ks))
        else:
            stride_mp4, _, stride_gif, _ = _strides(cbf.dt)
            ks_mp4 = set(range(k0, k1, stride_mp4)) if mp4 else set()
            ks_gif = set(range(k0, k1, stride_gif)) if gif else set()
            ks = sorted(ks_mp4 | ks_gif)
            if not ks:
                continue
        frames = _composites(
            spec["runs"],
            spec["labels"],
            ks,
            panel_wh=spec["panel_wh"],
            hud_wh=spec["hud_wh"],
            hud_run=cbf,
            compare_run=spec["compare"],
            downscale=spec["downscale"],
            ylim=_hud_ylim(cbf, k0, k1),
            offline=offline,
            floor=floor,
            k0=k0,
            k1=k1,
            need_hud=stills or mp4 or not gif_no_hud,
            plain_too=gif_no_hud and gif and not stills,
            drift=0.0 if no_drift else DRIFT_DEG_PER_S,
        )
        if stills:
            outputs += _write_stills(frames, out_base, still_ks, STILL_FRACTIONS)
        else:
            outputs += _encode(
                frames,
                out_base,
                cbf.dt,
                ks_mp4,
                ks_gif,
                mp4,
                gif,
                gif_width=gif_width,
                gif_colors=gif_colors,
                gif_fps=gif_fps,
            )
    return outputs


def main(argv=None) -> List[str]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--npz", required=True, help="the filtered run written by `kuka_cell.py simulate`"
    )
    p.add_argument("--unfiltered-npz", default=None, help="the comparison run (--side-by-side)")
    p.add_argument("--out", default=SHOWCASE_DIR, help=f"output directory (default {SHOWCASE_DIR})")
    p.add_argument("--side-by-side", action="store_true", help="also write the two-panel clips")
    p.add_argument("--stills", action="store_true", help="five review PNGs instead of video")
    p.add_argument("--max-seconds", type=float, default=None, help="truncate (smoke runs)")
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    g = p.add_argument_group("GIF size")
    g.add_argument("--gif-width", type=int, default=GIF_WIDTH)
    g.add_argument("--gif-colors", type=int, default=GIF_COLORS)
    g.add_argument("--gif-fps", type=float, default=GIF_FPS_OUT)
    g.add_argument("--gif-no-hud", action="store_true", help="GIF without the HUD strip")
    g.add_argument("--no-drift", action="store_true", help="hold the camera azimuth still")
    g.add_argument("--floor", choices=showcase.FLOOR_STYLES, default="grid")
    args = p.parse_args(argv)
    outputs = render(
        args.npz,
        unfiltered_npz=args.unfiltered_npz,
        out_dir=args.out,
        side_by_side=args.side_by_side,
        mp4=not args.no_mp4,
        gif=not args.no_gif,
        stills=args.stills,
        max_seconds=args.max_seconds,
        gif_width=args.gif_width,
        gif_colors=args.gif_colors,
        gif_fps=args.gif_fps,
        gif_no_hud=args.gif_no_hud,
        no_drift=args.no_drift,
        floor=args.floor,
    )
    for path in outputs:
        print(path)
    return outputs


if __name__ == "__main__":
    main()
