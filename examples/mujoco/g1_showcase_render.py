"""Render driver for the G1 showcase clips: npz in, MP4 / GIF / stills out.

``examples/mujoco/g1_showcase.py simulate <example>`` writes one npz per run; this module
replays it through a *render-only* model (studio lighting, identical ``nq``/``nv``), draws
the certificate overlays into the scene, composites the HUD strip underneath and encodes
the result. Nothing is re-simulated here, so a run can be re-cut as often as the look needs
changing.

    python examples/mujoco/g1_showcase_render.py navigate --npz results/showcase/g1_navigate.npz
    python examples/mujoco/g1_showcase_render.py corridor --npz .../g1_corridor.npz \\
        --unfiltered-npz .../g1_corridor_unfiltered.npz --side-by-side
    python examples/mujoco/g1_showcase_render.py scramble --npz ... --stills --max-seconds 8

What lands in ``--out``:

* ``g1_<example>.mp4``  -- one 1280x720 panel + a 1280x160 HUD, real time (25 fps).
* ``g1_<example>.gif``  -- the same shot at 2x speed, 20 fps, 480 px wide, 64 colours.
* ``g1_<example>_side_by_side.{mp4,gif}`` with ``--side-by-side``: the unfiltered run on the
  left and the filtered run on the right, both 960x540, under a 1920x180 HUD that carries
  the filtered run's readouts plus the unfiltered ``h_min`` as a red dashed trace. The left
  label is per example (:data:`LABEL_UNFILTERED`): corridor and scramble keep their MPPI
  local planner when the certificates go, so only navigate and plaza are policy-only.
* ``g1_<example>_still_{000,025,050,075,100}.png`` with ``--stills``, at those fractions of
  the rendered stretch -- the review artefact, cheap enough to iterate the look on.

``--window T0 T1`` narrows every output to those simulated seconds, defaulting per example
to :data:`WINDOWS` (corridor and scramble open partway in, where the interesting crossing
is) and switched off entirely by ``--full-clip``. The HUD clock stays absolute.

Frame cadence. The logs are 50 Hz (``dt`` 0.02), so the MP4 takes every 2nd logged step at
25 fps (real time) and the GIF takes every 5th at 10 fps written out at 2x (20 fps, 10
frames per simulated second -- the cadence of the current README clips). Both strides are
derived from the npz's ``dt``, and the union of the two frame sets is rendered once.

Two notes on what the HUD shows:

* the intervention bar is the *filtered* variable, ``|a_safe - a_nom|`` for the
  double-integrator wrappers and ``|v_safe - v_nom|`` for the single-integrator one. The
  DI wrappers certify an acceleration and integrate it, so ``v_safe`` is an integral of the
  nominal and ``|v_safe - v_nom|`` is large even with no certificates at all (see the
  ``--unfiltered`` caveat in ``g1_showcase.py``); it would light "CBF ACTIVE" permanently.
* ``h`` is the npz's recomputed true barrier value, not ``bfs`` (which is psi_1 of the
  rectified high-order barrier). A run whose ``barrier_shape`` is the *quadratic* ellipsoid
  is plotted and coloured in the equivalent distance form ``sqrt(h + 1) - 1`` (same zero
  crossing, same sign), because ``(d/r)^2 - 1`` opens above 7 and flattens the whole trace;
  the plot says so in its title and the npz keeps the raw values.
"""

import argparse
import contextlib
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import mujoco
import numpy as np

from cbfkit.systems.mujoco import showcase
from cbfkit.systems.mujoco.viewer_utils import add_marker

EXAMPLES = ("navigate", "plaza", "corridor", "scramble")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SHOWCASE_DIR = os.path.join(RESULTS_DIR, "showcase")

# --------------------------------------------------------------------------- output format
PANEL_W, PANEL_H = 1280, 720
HUD_H = 160
SBS_PANEL_W, SBS_PANEL_H = 960, 540
SBS_HUD_W, SBS_HUD_H = 1920, 180
MAX_GEOM = 20000
FPS_MP4 = 25.0  # real time at dt = 0.02
FPS_GIF = 10.0  # x GIF_SPEED = 20 fps output
GIF_SPEED = 2.0
GIF_WIDTH = 480
GIF_COLORS = 64
STILL_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
TAIL_S = 1.0  # rendered past the goal hit, so the clip does not cut on the last step

# --------------------------------------------------------------------------- overlay look
ARROW_SCALE = 1.2  # m per (m/s): a 0.5 m/s command draws a 0.6 m arrow
ARROW_MIN_LEN = 0.15  # a slow command still has to be visible
ARROW_WIDTH = 0.05
ARROW_Z_NOM = 0.05  # floor arrows, not pelvis arrows: at torso height they vanish in the mesh
ARROW_Z_SAFE = 0.07  # above v_nom so the two do not z-fight when the filter is inactive
TRAIL_S = 6.0
TRAIL_POINTS = 60  # the trail is resampled to this many points (one capsule per segment)
RING_WIDTH = 0.02
GOAL_RING_WIDTH = 0.03
GOAL_SPHERE_R = 0.12  # the beacon marks the spot; the goal *region* is the floor rings
PLAN_WIDTH = 0.03
PLAN_Z = 0.02  # above the keep-out discs, which sit at 4 mm
GHOST_THICKNESS = 0.003
PED_BODY_SCALE = 0.8  # of ped_radius: at 0.6 the pedestrians read as thin poles
PED_HEIGHT = 1.65
# In the scramble there are 40 of them: at 0.8 the crowd screens the robot off entirely.
PED_BODY_SCALE_CROWD = 0.5
PED_HEIGHT_CROWD = 1.6
CROWD_CULL_M = 9.0  # a pedestrian farther than this from the CoM is not drawn at all
DASH_LEN = 0.25
N_RING_NEAR = 4  # scramble: a ring is 48 geoms, so only the nearest few get one
HEADING_TAU = 0.5  # s, smoothing of the camera heading when theta_cmd is absent

COL_V_NOM = (0.55, 0.55, 0.55, 0.9)
COL_V_SAFE = (0.20, 0.75, 0.35, 1.0)
COL_TRAIL = (0.20, 0.50, 0.95, 0.9)
COL_PLAN = (1.00, 0.85, 0.20, 0.95)
COL_OBSTACLE = (0.85, 0.15, 0.20, 1.0)
COL_PED = (0.20, 0.40, 0.90, 1.0)
COL_GOAL = (0.10, 0.80, 0.20, 0.6)
COL_GOAL_RING = (0.10, 0.80, 0.20, 0.9)
COL_GOAL_RING_INNER = (0.10, 0.80, 0.20, 0.5)
COL_ROUTE = (0.10, 0.80, 0.20, 0.55)
COL_GHOST = (0.10, 0.10, 0.15, 0.55)
COL_KERB = (0.60, 0.60, 0.60, 1.0)

# distance [m] / elevation [deg] per example
CAMERAS: Dict[str, Tuple[float, float]] = {
    "navigate": (3.2, -18.0),
    "plaza": (4.5, -24.0),
    "corridor": (3.6, -20.0),
    "scramble": (6.5, -40.0),  # steeper: look down into the crowd rather than through it
}

# Default ``--window`` per example, in simulated seconds: the stretch worth showing. None
# renders the whole live part. A window that falls outside a (short) run is dropped.
WINDOWS: Dict[str, Optional[Tuple[float, float]]] = {
    "navigate": None,
    "plaza": None,
    "corridor": (20.0, 56.0),
    "scramble": (6.0, 42.0),
}
WINDOW_DEFAULT = "example-default"

# `barrier_shape` values whose h is the *quadratic* form (d/r)^2 - 1 rather than d/r - 1.
_QUADRATIC_SHAPES = ("ellipsoid", "quadratic")
H_LABEL = "h_min"
H_LABEL_DISTANCE = "h_min (distance form)"
AZIMUTH0 = 135.0
DRIFT_DEG_PER_S = 0.6
LAG_S = 0.4

LABEL_CBF = "+ CBFKit safety filter"
# What the comparison run actually is, per example. ``corridor`` and ``scramble`` keep the
# MPPI local planner when the certificates are removed, so their left panel is "a planner
# without a certificate", not "the walking policy alone"; ``navigate`` and ``plaza`` fall
# back to their P-law nominals and are the honest policy-only contrast.
LABEL_UNFILTERED = {
    "navigate": "Walking policy only",
    "plaza": "Walking policy only",
    "corridor": "MPPI planner, no certificate",
    "scramble": "MPPI planner, no certificate",
}
# When the npz declares what was left driving the robot, that wins over the example map.
LABEL_BY_MODE = {
    "nominal": "Goal-directed policy, no filter",
    "planner": "MPPI planner, no certificate",
}

_CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
_MODEL_CACHE: Dict[Tuple[str, bool], mujoco.MjModel] = {}


# --------------------------------------------------------------------------- the npz
class Run:
    """One simulated run: the npz's arrays plus the few things every overlay needs.

    Optional keys are genuinely optional -- an unfiltered run has no ``bfs``/``violated``,
    a non-MPPI run no ``mppi_x_traj``, a disc-footprint run no ``theta_cmd`` -- so
    everything beyond the mandatory core is reached through :meth:`get`.
    """

    _REQUIRED = ("states", "dt", "plant_kind", "nq", "nv", "com", "n_live", "h")

    def __init__(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as handle:
            self.arrays: Dict[str, np.ndarray] = {k: handle[k] for k in handle.files}
        missing = [k for k in self._REQUIRED if k not in self.arrays]
        if missing:
            raise KeyError(f"{path} is missing showcase keys {missing}")
        self.path = str(path)
        self.states = np.asarray(self.arrays["states"], dtype=float)
        self.dt = float(self.arrays["dt"])
        self.plant_kind = str(self.arrays["plant_kind"])
        self.nq = int(self.arrays["nq"])
        self.nv = int(self.arrays["nv"])
        self.com = np.asarray(self.arrays["com"], dtype=float).reshape(-1, 2)
        self.n_live = int(self.arrays["n_live"])
        self.h = np.asarray(self.arrays["h"], dtype=float).reshape(len(self.states), -1)
        self.barrier_shape = str(self.arrays.get("barrier_shape", np.array("")))
        # navigate's barrier is the quadratic ellipsoid (d/r)^2 - 1, which opens above 7 and
        # flattens everything near zero on a shared plot. Display it in the distance form
        # d/r - 1 that the other three already use: same zero crossing, same sign, readable.
        self.distance_form = self.barrier_shape in _QUADRATIC_SHAPES
        self.h_display = (
            np.sqrt(np.maximum(self.h + 1.0, 0.0)) - 1.0 if self.distance_form else self.h
        )
        self.h_min = self.h_display.min(axis=1) if self.h.shape[1] else np.zeros(len(self.states))
        self.T = int(self.states.shape[0])
        self.t = np.arange(self.T, dtype=float) * self.dt
        self.example = str(self.arrays["example"]) if "example" in self.arrays else ""
        self.unfiltered = bool(self.arrays.get("unfiltered", np.bool_(False)))
        self.n_u = int(self.arrays["n_u"]) if "n_u" in self.arrays else 0
        self.relax = bool(self.arrays["relax"]) if "relax" in self.arrays else False

    def get(self, key: str) -> Optional[np.ndarray]:
        """The array under ``key``, or None when this run did not log it."""
        val = self.arrays.get(key)
        return None if val is None else np.asarray(val)

    def scalar(self, key: str, default: float) -> float:
        val = self.arrays.get(key)
        return default if val is None else float(val)

    def points(self, key: str) -> np.ndarray:
        """An ``(n, 2)`` metadata array (obstacles, waypoints), possibly empty."""
        val = self.arrays.get(key)
        if val is None:
            return np.zeros((0, 2))
        return np.asarray(val, dtype=float).reshape(-1, 2)

    def radii(self, key: str, n: int, default: float) -> np.ndarray:
        val = self.arrays.get(key)
        if val is None:
            return np.full(n, float(default))
        return np.asarray(val, dtype=float).reshape(-1)

    # -- derived ----------------------------------------------------------
    @property
    def goal(self) -> np.ndarray:
        val = self.arrays.get("goal")
        return self.com[-1] if val is None else np.asarray(val, dtype=float).reshape(-1)[:2]

    @property
    def goal_radius(self) -> float:
        return self.scalar("goal_radius", 0.3)

    @property
    def ped_radius(self) -> float:
        return self.scalar("ped_radius", 0.3)

    @property
    def ped_keepout(self) -> float:
        return self.scalar("ped_keepout", self.ped_radius + self.scalar("robot_radius", 0.35))

    @property
    def footprint_axes(self) -> np.ndarray:
        val = self.arrays.get("footprint_axes")
        r = self.scalar("robot_radius", 0.35)
        return np.array([r, r]) if val is None else np.asarray(val, dtype=float).reshape(-1)[:2]

    def clip(self, k: int) -> int:
        """``k`` clamped into this run's logged range (a shorter run freezes on its last state)."""
        return int(min(max(int(k), 0), self.T - 1))


def _intervention(run: Run) -> Tuple[np.ndarray, str]:
    """Per-step ``|u_safe - u_nom|`` of the variable the QP actually filters, and its caption.

    The DI/HDI wrappers certify an acceleration and integrate it into ``v_safe``, so
    ``|v_safe - v_nom|`` is the tracking error of a double integrator, not an intervention;
    ``a`` is the honest choice wherever it was logged.
    """
    for nom_key, safe_key, symbol, unit in (
        ("a_nom", "a_safe", "da", "m/s^2"),
        ("v_nom", "v_safe", "dv", "m/s"),
    ):
        nom, safe = run.get(nom_key), run.get(safe_key)
        if nom is None or safe is None:
            continue
        nom = np.asarray(nom, dtype=float).reshape(run.T, -1)
        safe = np.asarray(safe, dtype=float).reshape(run.T, -1)
        n = min(nom.shape[1], safe.shape[1])
        return np.linalg.norm(safe[:, :n] - nom[:, :n], axis=1), f"|{symbol}| = {{:.2f}} {unit}"
    return np.zeros(run.T), "|dv| = {:.2f} m/s"


def _heading(run: Run) -> np.ndarray:
    """Per-step walking direction for the camera: ``theta_cmd`` when logged, else smoothed v."""
    theta = run.get("theta_cmd")
    if theta is not None:
        return np.asarray(theta, dtype=float).reshape(-1)
    vel = run.get("v_safe")
    if vel is None:
        vel = np.gradient(run.com, run.dt, axis=0) if run.T > 1 else np.zeros((run.T, 2))
    vel = np.asarray(vel, dtype=float).reshape(run.T, -1)[:, :2]
    alpha = 1.0 - float(np.exp(-run.dt / HEADING_TAU))
    smooth = np.zeros_like(vel)
    acc = vel[0].copy()
    for i in range(run.T):
        acc = acc + alpha * (vel[i] - acc)
        smooth[i] = acc
    out = np.zeros(run.T)
    last = float(np.arctan2(smooth[0, 1], smooth[0, 0])) if run.T else 0.0
    for i in range(run.T):
        if float(np.linalg.norm(smooth[i])) > 1e-3:
            last = float(np.arctan2(smooth[i, 1], smooth[i, 0]))
        out[i] = last
    return out


# --------------------------------------------------------------------------- overlay bits
def _person(
    scn: mujoco.MjvScene,
    p_xy: Sequence[float],
    v_xy: Sequence[float],
    r: float,
    body_scale: float = PED_BODY_SCALE,
    height: float = PED_HEIGHT,
) -> int:
    """One pedestrian at their own footprint radius; the keep-out boundary is drawn apart."""
    return showcase.pedestrian(scn, p_xy, v_xy, r, COL_PED, height=height, body_scale=body_scale)


def _keepout(scn: mujoco.MjvScene, xy: Sequence[float], r: float, h_val: float) -> None:
    """The constraint boundary of one barrier: a translucent disc plus a ring, coloured by h."""
    showcase.disc(scn, xy, r, showcase.h_rgba(h_val, 0.30))
    showcase.ring(scn, xy, r, showcase.h_rgba(h_val, 0.95), width=RING_WIDTH)


def _dashed(
    scn: mujoco.MjvScene,
    points_xy: np.ndarray,
    rgba: Sequence[float],
    dash: float = DASH_LEN,
    width: float = 0.03,
) -> int:
    """A dashed floor polyline: every other ``dash``-long piece of each segment is drawn."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    added = 0
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        n = max(2, int(round(length / dash)))
        fractions = np.linspace(0.0, 1.0, n + 1)[:, None]
        seg = a + fractions * (b - a)
        for i in range(0, n, 2):
            added += showcase.path(scn, seg[i : i + 2], rgba, width=width)
    return added


def _arrows(scn: mujoco.MjvScene, run: Run, k: int) -> None:
    """``v_nom`` (gray) and ``v_safe`` (green) as floor arrows from the CoM ground point."""
    p = run.com[k]
    for key, colour, z in (
        ("v_nom", COL_V_NOM, ARROW_Z_NOM),
        ("v_safe", COL_V_SAFE, ARROW_Z_SAFE),
    ):
        vel = run.get(key)
        if vel is None:
            continue
        v = np.asarray(vel[k], dtype=float).reshape(-1)[:2]
        speed = float(np.linalg.norm(v))
        if speed < 1e-3:  # no direction to point in
            continue
        tip = p + (v / speed) * max(ARROW_MIN_LEN, ARROW_SCALE * speed)
        showcase.arrow(scn, (p[0], p[1], z), (tip[0], tip[1], z), colour, width=ARROW_WIDTH)


def _trail_and_plan(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    """The last 6 s of CoM (blue, fading) and the current MPPI plan (amber)."""
    lo = max(0, k - prep["trail_steps"])
    pts = run.com[lo : k + 1]
    if pts.shape[0] > TRAIL_POINTS:
        idx = np.linspace(0, pts.shape[0] - 1, TRAIL_POINTS).round().astype(int)
        pts = pts[idx]
    showcase.trail(scn, pts, COL_TRAIL)
    plan = run.get("mppi_x_traj")
    if plan is not None and plan.ndim == 3:
        showcase.path(
            scn, np.asarray(plan[k], dtype=float)[:2].T, COL_PLAN, width=PLAN_WIDTH, z=PLAN_Z
        )


def _common(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    _trail_and_plan(scn, run, k, prep)
    _arrows(scn, run, k)


def _goal(scn: mujoco.MjvScene, run: Run, pos_xy: Optional[Sequence[float]] = None) -> None:
    """The goal: a small beacon on the spot, and the goal *region* as two floor rings."""
    p = run.goal if pos_xy is None else np.asarray(pos_xy, dtype=float).reshape(-1)[:2]
    showcase.ring(scn, p, run.goal_radius, COL_GOAL_RING, width=GOAL_RING_WIDTH)
    showcase.ring(scn, p, 0.5 * run.goal_radius, COL_GOAL_RING_INNER, width=GOAL_RING_WIDTH)
    showcase.beacon(scn, p, GOAL_SPHERE_R, COL_GOAL)


def _pillars(scn: mujoco.MjvScene, run: Run, k: int, half_height: float) -> int:
    """Static obstacles as solid red cylinders with their keep-out boundary; returns how many."""
    obstacles = run.points("obstacles")
    radii = run.radii("obstacle_radii", len(obstacles), 0.3)
    keepout = run.radii("keepout_radii", len(obstacles), 0.65)
    for i, p in enumerate(obstacles):
        add_marker(scn, _CYL, [radii[i], half_height, 0], [p[0], p[1], half_height], COL_OBSTACLE)
        if i < run.h_display.shape[1]:
            _keepout(scn, p, keepout[i], float(run.h_display[k, i]))
    return len(obstacles)


def _crowd(
    scn: mujoco.MjvScene,
    run: Run,
    k: int,
    *,
    h_offset: int,
    rings: bool,
    n_rings: Optional[int] = None,
    cull_m: Optional[float] = None,
    body_scale: float = PED_BODY_SCALE,
    height: float = PED_HEIGHT,
) -> None:
    """The logged pedestrians: a body each, plus the keep-out disc/ring of the nearest ones.

    ``h_offset`` is where the pedestrian columns start in ``h`` (after the static obstacles).
    ``n_rings`` caps how many rings are drawn -- a ring is 48 geoms, so in the scramble only
    the nearest few get one. ``cull_m`` drops anyone farther than that from the CoM: in a
    40-person crowd the distant half only screens off the robot the shot is about.
    """
    agents = run.get("agents")
    if agents is None or agents.ndim != 3:
        return
    state = np.asarray(agents[k], dtype=float)
    pos, vel = state[:, :2], state[:, 2:4]
    distance = np.linalg.norm(pos - run.com[k], axis=1)
    order = range(len(pos))
    if rings and n_rings is not None and len(pos) > n_rings:
        order = np.argsort(distance)[:n_rings].tolist()
    ringed = set(order) if rings else set()
    for j in range(len(pos)):
        if cull_m is not None and float(distance[j]) > cull_m:
            continue
        _person(scn, pos[j], vel[j], run.ped_radius, body_scale=body_scale, height=height)
        col = h_offset + j
        if j in ringed and col < run.h_display.shape[1]:
            _keepout(scn, pos[j], run.ped_keepout, float(run.h_display[k, col]))


# --------------------------------------------------------------------------- per example
def _overlay_navigate(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    """One obstacle cylinder with its keep-out ring, the goal beacon, arrows and trail."""
    _pillars(scn, run, k, half_height=0.5)
    _goal(scn, run)
    _common(scn, run, k, prep)


def _overlay_plaza(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    """Pillars, three pedestrians, the dashed waypoint route and its active waypoint.

    This is the only overlay that reads ``x_traj``: plaza's is the route planner's *active*
    waypoint, while corridor's and scramble's are a constant goal that would just duplicate
    the ``goal`` beacon.
    """
    n_obs = _pillars(scn, run, k, half_height=0.6)
    _crowd(scn, run, k, h_offset=n_obs, rings=True)
    waypoints = run.points("waypoints")
    if len(waypoints):
        _dashed(scn, np.vstack([run.com[0], waypoints]), COL_ROUTE)
    active = run.get("x_traj")
    target = (
        np.asarray(active[k], dtype=float).reshape(-1)[:2]
        if active is not None and active.ndim == 3
        else run.goal
    )
    _goal(scn, run, target)
    _common(scn, run, k, prep)


def _overlay_corridor(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    """The two static pedestrians and the rotating footprint ellipse that has to thread them.

    The constraint boundary here is the ellipse *around the robot* (semi-axes already
    inflated by the pedestrian radius), so the pedestrians carry no separate keep-out ring;
    the thin gray ghost is the same ellipse at the realised pelvis yaw, which is how far the
    robot's body is from the heading the certificate assumed.
    """
    if run.get("agents") is None:  # the pedestrians are motionless; the metadata has them
        for p in run.points("obstacles"):
            _person(scn, p, (0.0, 0.0), run.ped_radius)
    else:
        _crowd(scn, run, k, h_offset=0, rings=False)
    axes = run.get("footprint_axes_inflated")
    axes = run.footprint_axes + run.ped_radius if axes is None else np.asarray(axes, dtype=float)
    theta = run.get("theta_cmd")
    theta_k = float(theta[k]) if theta is not None else float(prep["heading"][k])
    showcase.ellipse(
        scn, run.com[k], axes[0], axes[1], theta_k, showcase.h_rgba(float(run.h_min[k]), 0.45)
    )
    showcase.ellipse(
        scn,
        run.com[k],
        axes[0],
        axes[1],
        float(prep["pelvis_yaw"][k]),
        COL_GHOST,
        z=0.0075,
        thickness=GHOST_THICKNESS,
    )
    _goal(scn, run)
    _common(scn, run, k, prep)


def _overlay_scramble(scn: mujoco.MjvScene, run: Run, k: int, prep: Dict[str, Any]) -> None:
    """The nearby crowd, rings for the nearest four, the kerb markers and the goal."""
    _crowd(
        scn,
        run,
        k,
        h_offset=0,
        rings=True,
        n_rings=N_RING_NEAR,
        cull_m=CROWD_CULL_M,
        body_scale=PED_BODY_SCALE_CROWD,
        height=PED_HEIGHT_CROWD,
    )
    half = run.arrays.get("half")
    if half is not None:
        h = float(half)
        for cx, cy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            add_marker(scn, _CYL, [0.12, 0.3, 0], [cx * h, cy * h, 0.3], COL_KERB)
    _goal(scn, run)
    _common(scn, run, k, prep)


OVERLAYS = {
    "navigate": _overlay_navigate,
    "plaza": _overlay_plaza,
    "corridor": _overlay_corridor,
    "scramble": _overlay_scramble,
}


def _prepare(run: Run, example: str) -> Dict[str, Any]:
    """Everything an overlay needs that is cheaper computed once than per frame."""
    prep: Dict[str, Any] = {
        "trail_steps": max(1, int(round(TRAIL_S / run.dt))),
        "heading": _heading(run),
    }
    if example == "corridor":
        prep["pelvis_yaw"] = showcase.pelvis_yaw(run.states, run.nq)
    return prep


# --------------------------------------------------------------------------- rendering
def _model(plant_kind: str, offline: bool) -> mujoco.MjModel:
    key = (plant_kind, bool(offline))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = showcase.render_model(plant_kind, offline=offline)
        _MODEL_CACHE[key] = model
    return model


class _Panel:
    """One 3-D view: a render-only model, its own ``MjData``/``Renderer``, and an overlay."""

    def __init__(self, run: Run, example: str, width: int, height: int, offline: bool) -> None:
        self.run = run
        self.model = _model(run.plant_kind, offline)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width, max_geom=MAX_GEOM)
        self.overlay = OVERLAYS[example]
        self.prep = _prepare(run, example)

    def frame(self, k: int, cam: mujoco.MjvCamera) -> np.ndarray:
        kk = self.run.clip(k)
        nq, nv = self.run.nq, self.run.nv
        self.data.qpos[:] = self.run.states[kk, :nq]
        self.data.qvel[:] = self.run.states[kk, nq : nq + nv]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=cam)
        self.overlay(self.renderer.scene, self.run, kk, self.prep)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def _unfiltered_label(run: Run, example: str) -> str:
    """What the comparison panel actually shows.

    ``unfiltered_mode`` in the npz says what was left driving the robot once the certificates
    were removed -- ``"nominal"`` for the P-law goal seeker, ``"planner"`` when the MPPI local
    planner is still in the loop. Older npz files predate the key, so the per-example map
    stands in for them.
    """
    mode = run.arrays.get("unfiltered_mode")
    if mode is not None:
        return LABEL_BY_MODE.get(str(mode), LABEL_UNFILTERED[example])
    return LABEL_UNFILTERED[example]


def _mode(run: Run, example: str) -> str:
    """The HUD's small label under the clock: what is producing the command at all."""
    if run.unfiltered:
        return _unfiltered_label(run, example).lower()
    return "CBF-QP + MPPI" if run.get("mppi_x_traj") is not None else "CBF-QP"


def _hud_kwargs(run: Run, example: str, k: int, iv: np.ndarray, caption: str) -> Dict[str, Any]:
    kk = run.clip(k)
    flags: Dict[str, bool] = {}
    mppi_error = run.get("mppi_error")
    if mppi_error is not None:
        flags["MPPI fallback"] = bool(np.asarray(mppi_error).reshape(-1)[kk])
    status = run.get("solver_status")
    if status is not None:
        flags["QP not converged"] = int(np.asarray(status).reshape(-1)[kk]) != 1
    # A relaxed run that had its certificates removed has no slack columns at all, so `sol`
    # is (T, n_u) there rather than (T, n_u + n_barriers) and the pill is simply absent.
    slack: Optional[float] = None
    sol = run.get("sol")
    if run.relax and sol is not None and sol.ndim == 2 and sol.shape[1] > run.n_u:
        slack = float(np.max(np.asarray(sol[kk], dtype=float)[run.n_u :]))
    value = float(iv[kk])
    return dict(
        t=float(run.t[kk]),
        h_min=float(run.h_min[kk]),
        h_hist=run.h_min[: kk + 1],
        t_hist=run.t[: kk + 1],
        intervention=value,
        active=value > 1e-3,
        flags=flags,
        slack=slack,
        mode=_mode(run, example),
        intervention_caption="intervention  " + caption.format(value),
        h_label=H_LABEL_DISTANCE if run.distance_form else H_LABEL,
    )


def _composites(
    runs: List[Run],
    labels: List[str],
    example: str,
    ks: Sequence[int],
    *,
    panel_wh: Tuple[int, int],
    hud_wh: Tuple[int, int],
    cam_run: Run,
    hud_run: Run,
    compare_run: Optional[Run],
    downscale: Optional[Tuple[int, int]],
    ylim: Tuple[float, float],
    offline: bool,
):
    """Yield ``(k, composite RGB frame)`` for each ``k``, rendering each frame exactly once.

    The camera schedule is driven by ``cam_run`` for every panel so the two side-by-side
    shots stay comparable, and the HUD carries ``hud_run``'s readouts with ``compare_run``'s
    ``h_min`` as the second trace.
    """
    from PIL import Image

    distance, elevation = CAMERAS[example]
    panels = [_Panel(r, example, panel_wh[0], panel_wh[1], offline) for r in runs]
    hud = showcase.HudRenderer(*hud_wh)
    cam = showcase.CameraSchedule(
        distance,
        elevation,
        azimuth0=AZIMUTH0,
        drift_deg_per_s=DRIFT_DEG_PER_S,
        lag_s=LAG_S,
        dt=cam_run.dt,
    )
    heading = _heading(cam_run)
    iv, caption = _intervention(hud_run)
    try:
        for k in ks:
            kk = cam_run.clip(k)
            camera = cam.update(k, cam_run.com[kk], heading=float(heading[kk]))
            images = [p.frame(k, camera) for p in panels]
            if downscale is not None:
                images = [
                    np.asarray(Image.fromarray(im).resize(downscale, Image.LANCZOS))
                    for im in images
                ]
            kwargs = _hud_kwargs(hud_run, example, k, iv, caption)
            kwargs["ylim"] = ylim
            if compare_run is not None:
                kk2 = compare_run.clip(k)
                kwargs["h_hist2"] = compare_run.h_min[: kk2 + 1]
            yield k, showcase.compose_panels(images, labels, hud.draw(**kwargs))
    finally:
        for panel in panels:
            panel.close()
        hud.close()


def _hud_ylim(run: Run, k0: int, k1: int) -> Tuple[float, float]:
    """One vertical range for the whole clip, so the trace does not jump frame to frame.

    The 90th percentile rather than the maximum: these barriers open far from every obstacle
    and letting that set the top flattens the part of the trace that matters, around zero.
    """
    segment = run.h_min[k0:k1]
    if segment.size == 0:
        segment = run.h_min
    lo = min(-0.25, 1.1 * float(segment.min()))
    hi = max(1.0, float(np.percentile(segment, 90.0)))
    return lo, hi


def _live_length(runs: Sequence[Run]) -> int:
    """Logged steps of the live part: the longest, plus a second, inside every run's log."""
    dt = runs[0].dt
    n = max(int(r.n_live) for r in runs) + int(round(TAIL_S / dt)) + 1
    return max(1, min([n] + [int(r.T) for r in runs]))


def _window_indices(
    n: int, dt: float, window: Optional[Tuple[float, float]], max_seconds: Optional[float]
) -> Tuple[int, int]:
    """The ``[k0, k1)`` logged steps to render: the live part, narrowed by window and cap.

    A window that starts past the end of a run (a smoke run, or a crossing that finished
    early) is dropped with a note rather than producing an empty clip.
    """
    k0, k1 = 0, n
    if window is not None:
        a, b = int(round(float(window[0]) / dt)), int(round(float(window[1]) / dt))
        if 0 <= a < n and b > a:
            k0, k1 = a, min(n, b)
        else:
            print(
                f"note: --window {window[0]:g} {window[1]:g} s lies outside the "
                f"{n * dt:.1f} s live part; rendering all of it"
            )
    if max_seconds is not None:
        k1 = min(k1, k0 + max(1, int(round(float(max_seconds) / dt))))
    return k0, max(k1, k0 + 1)


def _strides(dt: float) -> Tuple[int, float, int, float]:
    stride_mp4 = max(1, int(round((1.0 / FPS_MP4) / dt)))
    stride_gif = max(1, int(round((1.0 / FPS_GIF) / dt)))
    return stride_mp4, 1.0 / (dt * stride_mp4), stride_gif, 1.0 / (dt * stride_gif)


def _encode(
    frames,
    out_base: str,
    dt: float,
    ks_mp4: set,
    ks_gif: set,
    mp4: bool,
    gif: bool,
) -> List[str]:
    """Push each composite into whichever writers asked for that frame index.

    The two frame sets overlap only every 10th step, so the stack renders the union once
    and each writer takes the frames it wants. Leaving the stack encodes both files; an
    exception on the way tears the ffmpeg pipes down instead of leaving them running.
    """
    _, fps_mp4, _, fps_gif = _strides(dt)
    outputs: List[str] = []
    with contextlib.ExitStack() as stack:
        mp4_writer = None
        if mp4:
            mp4_writer = stack.enter_context(
                showcase.FrameWriter(f"{out_base}.mp4", fps_mp4, kind="mp4")
            )
            outputs.append(f"{out_base}.mp4")
        gif_writer = None
        if gif:
            gif_writer = stack.enter_context(
                showcase.FrameWriter(
                    f"{out_base}.gif",
                    fps_gif,
                    kind="gif",
                    speed=GIF_SPEED,
                    gif_width=GIF_WIDTH,
                    gif_colors=GIF_COLORS,
                )
            )
            outputs.append(f"{out_base}.gif")
        for k, frame in frames:
            if mp4_writer is not None and k in ks_mp4:
                mp4_writer.add(frame)
            if gif_writer is not None and k in ks_gif:
                gif_writer.add(frame)
    return outputs


def _write_stills(
    frames, out_base: str, ks: Sequence[int], fractions: Sequence[float]
) -> List[str]:
    """One PNG per ``(fraction, k)`` pair; ``ks`` may repeat when the live part is very short."""
    from PIL import Image

    outputs = []
    by_k = {k: frame for k, frame in frames}
    for fraction, k in zip(fractions, ks):
        path = f"{out_base}_still_{int(round(fraction * 100)):03d}.png"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        Image.fromarray(by_k[k]).save(path)
        outputs.append(path)
    return outputs


# --------------------------------------------------------------------------- public API
def render(
    example: str,
    npz: str,
    *,
    unfiltered_npz: Optional[str] = None,
    out_dir: str = SHOWCASE_DIR,
    side_by_side: bool = False,
    mp4: bool = True,
    gif: bool = True,
    stills: bool = False,
    max_seconds: Optional[float] = None,
    window: Union[Tuple[float, float], str, None] = WINDOW_DEFAULT,
) -> List[str]:
    """Render ``npz`` (and, with ``side_by_side``, its unfiltered twin) and return the paths.

    ``example`` selects the overlay set and the camera framing; it must be one of
    :data:`EXAMPLES`. ``npz`` is the filtered run written by ``g1_showcase.py simulate``.
    ``stills`` replaces the video outputs with five review PNGs; ``max_seconds`` truncates
    the clip (smoke runs). ``mp4``/``gif`` turn off one encoder or the other.

    ``window`` is ``(t0, t1)`` in simulated seconds and applies to every output; it defaults
    to :data:`WINDOWS` for the example, and ``None`` renders the whole live part. The HUD
    clock stays absolute, so a windowed clip opens partway through the run.

    With ``side_by_side`` the single-panel outputs are written first and the two-panel pair
    additionally, both driven by the filtered run's camera so the shots stay comparable.
    """
    if example not in EXAMPLES:
        raise ValueError(f"unknown example {example!r}; expected one of {EXAMPLES}")
    offline = bool(os.environ.get("CBFKIT_ASSETS_OFFLINE"))
    cbf = Run(npz)
    if cbf.example and cbf.example != example:
        raise ValueError(f"{npz} was simulated as {cbf.example!r}, not {example!r}")
    unfiltered: Optional[Run] = None
    if unfiltered_npz is not None:
        unfiltered = Run(unfiltered_npz)
        if abs(unfiltered.dt - cbf.dt) > 1e-12:
            raise ValueError(f"dt mismatch: {cbf.dt} vs {unfiltered.dt}")
    if side_by_side and unfiltered is None:
        raise ValueError("side_by_side needs --unfiltered-npz")
    if isinstance(window, str):
        if window != WINDOW_DEFAULT:
            raise ValueError(f"window must be a (t0, t1) pair or None, got {window!r}")
        window = WINDOWS[example]

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
                labels=[_unfiltered_label(unfiltered, example), LABEL_CBF],
                panel_wh=(PANEL_W, PANEL_H),
                hud_wh=(SBS_HUD_W, SBS_HUD_H),
                downscale=(SBS_PANEL_W, SBS_PANEL_H),
                compare=unfiltered,
                length_runs=[cbf, unfiltered],
            )
        )

    for spec in passes:
        n = _live_length(spec["length_runs"])
        k0, k1 = _window_indices(n, cbf.dt, window, max_seconds)
        out_base = os.path.join(out_dir, f"g1_{example}{spec['suffix']}")
        if stills:
            # without a window the stills stop at the goal hit, not in the frozen tail
            hi = k1 if window is not None else min(k1, max(k0 + 1, int(cbf.n_live)))
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
            example,
            ks,
            panel_wh=spec["panel_wh"],
            hud_wh=spec["hud_wh"],
            cam_run=cbf,
            hud_run=cbf,
            compare_run=spec["compare"],
            downscale=spec["downscale"],
            ylim=_hud_ylim(cbf, k0, k1),
            offline=offline,
        )
        if stills:
            outputs += _write_stills(frames, out_base, still_ks, STILL_FRACTIONS)
        else:
            outputs += _encode(frames, out_base, cbf.dt, ks_mp4, ks_gif, mp4, gif)
    return outputs


def main(argv=None) -> List[str]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("example", choices=EXAMPLES)
    p.add_argument("--npz", required=True, help="the filtered run written by `g1_showcase.py`")
    p.add_argument("--unfiltered-npz", default=None, help="the comparison run (--side-by-side)")
    p.add_argument("--out", default=SHOWCASE_DIR, help=f"output directory (default {SHOWCASE_DIR})")
    p.add_argument("--side-by-side", action="store_true", help="also write the two-panel clips")
    p.add_argument("--stills", action="store_true", help="five review PNGs instead of video")
    p.add_argument("--max-seconds", type=float, default=None, help="truncate (smoke runs)")
    p.add_argument(
        "--window",
        nargs=2,
        type=float,
        default=None,
        metavar=("T0", "T1"),
        help=f"simulated seconds to render (default per example: {WINDOWS})",
    )
    p.add_argument("--full-clip", action="store_true", help="ignore the example's default --window")
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args(argv)
    if a.window is not None:
        window: Union[Tuple[float, float], str, None] = (a.window[0], a.window[1])
    elif a.full_clip:
        window = None
    else:
        window = WINDOW_DEFAULT
    paths = render(
        a.example,
        a.npz,
        unfiltered_npz=a.unfiltered_npz,
        out_dir=a.out,
        side_by_side=a.side_by_side,
        mp4=not a.no_mp4,
        gif=not a.no_gif,
        stills=a.stills,
        max_seconds=a.max_seconds,
        window=window,
    )
    for path in paths:
        print(f"wrote {path}")
    return paths


if __name__ == "__main__":
    main()
