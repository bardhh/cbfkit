"""KUKA iiwa14 palletising cell with a worker crossing the arm's sweep.

A fixed-base industrial arm on a pedestal moves parts from a bin to a conveyor while a
worker walks through the transfer arc. The scene is assembled at runtime from the
menagerie ``kuka_iiwa_14`` model with :func:`build_spec`; the worker is a kinematic
``mocap`` body so MuJoCo resolves arm-worker contact physically -- the unfiltered arm
really strikes the mannequin.

Layers, from the plant up:

* :func:`make_plant` -- :class:`CellPlant`, a :class:`~cbfkit.systems.mujoco.plant.MujocoPlant`
  whose input is ``u = [qdot_cmd (7) | worker_xyz (3) | worker_quat (4)]``. The
  position servos of the menagerie model track ``qpos + qdot_cmd * dt`` and the worker
  pose is written to ``mocap_pos``/``mocap_quat`` every substep.
* :func:`worker_state` -- the worker script, a pure function of time.
* :func:`arm_worker_barriers` -- one CBF per (arm collision sphere, worker capsule) pair,
  ``h = dist - r_i - r_j - d_0 - tau |v_worker|``, differentiable through ``mjx.kinematics``.
* :func:`osc_nominal` -- resolved-rate operational-space pick-and-place cycle.
* :func:`safe_arm_controller` / :func:`build` -- the CBF-QP on commanded joint velocities
  wrapped for ``simulator.execute(plant=...)``.
"""

from typing import Any, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array
from mujoco import mjx

from cbfkit.certificates import certificate_package, concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.optimization.quadratic_program.solver_registry import get_solver
from cbfkit.systems.mujoco.plant import MujocoPlant
from cbfkit.utils.user_types import (
    CertificateCollection,
    ControllerCallable,
    ControllerData,
    DynamicsCallable,
)

__all__ = [
    "N_ARM",
    "NU",
    "PEDESTAL_H",
    "TOOL_OFFSET",
    "TOOL_R",
    "BIN_XY",
    "CONVEYOR_XY",
    "WORKER_X",
    "WORKER_Y0",
    "WORKER_Y_PAUSE",
    "WORKER_SPEED",
    "WORKER_T_ENTER",
    "WORKER_PAUSE_S",
    "WORKER_CAPSULES",
    "ARM_SPHERE_GEOMS",
    "D0",
    "TAU",
    "QD_MAX",
    "QD_LIMITS",
    "SLACK_PENALTY",
    "QP_SOLVER",
    "CLASS_K",
    "V_TOOL",
    "A_TOOL",
    "Q_HOME",
    "CYCLE_S",
    "H_NAMES",
    "worker_state",
    "worker_yaw",
    "worker_quat",
    "worker_capsules",
    "sphere_capsule_distance",
    "build_spec",
    "load_cell",
    "CellPlant",
    "make_plant",
    "arm_sphere_positions",
    "barrier_values",
    "arm_worker_barriers",
    "embedded_arm_worker",
    "tool_pose",
    "pick_place_reference",
    "osc_velocity",
    "phase_rate",
    "osc_nominal",
    "joint_limit_barriers",
    "safe_arm_controller",
    "build",
]

# --------------------------------------------------------------------------- layout (m, s)
N_ARM = 7
NU = 14
TIMESTEP = 0.002
SUBSTEPS = 10
CTRL_WINDUP = 0.15  # rad; servo target never runs further than this ahead of the joint
PEDESTAL_H = 0.30
PEDESTAL_R = 0.15
TOOL_OFFSET = 0.12  # gripper stand-in below the flange
TOOL_R = 0.07
BIN_XY = (0.40, -0.45)
BIN_HALF = (0.20, 0.20, 0.175)  # top at 0.35, spans x 0.20..0.60
CONVEYOR_XY = (0.45, 0.60)
CONVEYOR_HALF = (0.35, 0.20, 0.20)  # top at 0.40, spans x -0.10..0.80

# Worker path: enters walking +y along x = WORKER_X, pauses inside the sweep, walks back.
WORKER_X = 0.72  # torso surface at x 0.56, arms at 0.48: on the transfer path
WORKER_Y0 = -2.4
WORKER_Y_PAUSE = 0.15
WORKER_SPEED = 1.1
WORKER_T_ENTER = 1.0
WORKER_PAUSE_S = 1.5

# Capsules in the worker frame (+x = facing direction): (p0, p1, radius).
WORKER_CAPSULES: Tuple[
    Tuple[Tuple[float, float, float], Tuple[float, float, float], float], ...
] = (
    ((0.0, 0.0, 0.05), (0.0, 0.0, 0.85), 0.11),  # legs
    ((0.0, 0.0, 0.85), (0.0, 0.0, 1.40), 0.16),  # torso
    ((0.0, 0.0, 1.58), (0.0, 0.0, 1.58), 0.11),  # head
    ((0.0, 0.24, 1.35), (0.0, 0.24, 0.80), 0.05),  # left arm
    ((0.0, -0.24, 1.35), (0.0, -0.24, 0.80), 0.05),  # right arm
)
WORKER_CAPSULE_NAMES = ("legs", "torso", "head", "l_arm", "r_arm")
N_CAPSULES = len(WORKER_CAPSULES)

# Representative arm collision spheres: (body name, index among that body's collision
# geoms in the menagerie file). ``("link7", 1)`` is the tool sphere added by build_spec.
# The shoulder sphere on link2 is deliberately absent: it sits 0.2 m from the base axis and
# cannot get out of a walking worker's inflated keep-out, so a barrier on it makes the QP
# infeasible the moment the worker passes the pedestal (measured 2026-09-23). The pedestal
# is a fixed hazard the worker walks around; the arm's job is to keep its moving links clear.
ARM_SPHERE_GEOMS: Tuple[Tuple[str, int], ...] = (
    ("link3", 7),  # upper-arm tip
    ("link4", 5),  # elbow
    ("link4", 2),  # forearm mid
    ("link5", 0),  # forearm end
    ("link6", 2),  # wrist
    ("link7", 0),  # flange
    ("link7", 1),  # tool
)
ARM_SPHERE_NAMES = (
    "upperarm",
    "elbow",
    "forearm",
    "forearm2",
    "wrist",
    "flange",
    "tool",
)
N_SPHERES = len(ARM_SPHERE_GEOMS)
H_NAMES: Tuple[str, ...] = tuple(f"{a}/{c}" for a in ARM_SPHERE_NAMES for c in WORKER_CAPSULE_NAMES)

# Safety filter.
D0 = 0.10
TAU = 0.15
# iiwa14 datasheet joint speed limits (85, 85, 100, 75, 130, 135, 135 deg/s).
QD_LIMITS = (1.48, 1.48, 1.75, 1.31, 2.27, 2.36, 2.36)
QD_MAX = max(QD_LIMITS)
# Linear class-K gain. 1.0 (not 3.0) so the retreat starts about a metre out: the worker
# closes at 1.1 m/s and the base joint only turns at 1.48 rad/s, so a late start saturates
# the joints and leaves the QP infeasible.
CLASS_K = 1.0
# The CBF rows are soft with this penalty (the generator's default): a fixed-base arm can be
# walked into faster than it can fold away, and a hard QP then has no solution. With the
# penalty the QP always answers, and the slack is logged so a clip can show it.
SLACK_PENALTY = 2e3
# jaxopt's OSQP hits its 10 000-iteration cap on this QP once the arm is folded back and
# the box limits and several barrier rows are active together (measured 2026-09-23 at
# t = 2.7 s: status 5, relaxed or not); the registry's "fast" solver converges in ~16.
QP_SOLVER = "fast"
QP_TOL = 1e-8  # the fast solver's default 1e-6 leaves a 1e-3 rad/s residue on a pure projection
QP_MAX_ITER = 48  # the default 16 Newton steps cannot always reach 1e-8 (status 2); 48 does

# Nominal task.
V_TOOL = 1.2
A_TOOL = 3.0
KP_POS = 6.0
KP_ORI = 4.0
KP_NULL = 1.0
# The cycle clock only advances while the tool is within CATCHUP_R of its reference (a
# smooth ramp to zero over CATCHUP_BAND): a blocked arm holds its place in the cycle and
# resumes from there instead of chasing a reference that ran on without it.
CATCHUP_R = 0.30
CATCHUP_BAND = 0.15
JOINT_LIMIT_MARGIN = 0.05  # rad, joint-limit barriers h = (q_max - m) - q and q - (q_min + m)
DWELL_S = 0.3
Z_ABOVE_BIN = 0.80
Z_BIN = 0.47
Z_ABOVE_CONV = 0.85
Z_CONV = 0.52
# Joint configuration with the tool at bin_above pointing down (solved once with
# ``_solve_home`` below and frozen here so the example does not run IK at import).
Q_HOME = (-0.8350, 0.5044, -0.0153, -1.1332, 0.0074, 1.5040, -0.8000)


# --------------------------------------------------------------------------- worker script
def _walk_segment(t, t0, y0, y1, speed):
    """Position/velocity on a constant-speed walk from ``y0`` at ``t0`` to ``y1``."""
    direction = jnp.sign(y1 - y0)
    dur = jnp.abs(y1 - y0) / speed
    s = jnp.clip((t - t0) / dur, 0.0, 1.0)
    y = y0 + s * (y1 - y0)
    moving = (t >= t0) & (t < t0 + dur)
    v = jnp.where(moving, direction * speed, 0.0)
    return y, v, dur


def worker_state(t) -> Array:
    """Worker ``(px, py, vx, vy)`` at time ``t``: wait, walk in, pause, walk out, stand."""
    t = jnp.asarray(t, dtype=float)
    y_in, v_in, d_in = _walk_segment(t, WORKER_T_ENTER, WORKER_Y0, WORKER_Y_PAUSE, WORKER_SPEED)
    t_out = WORKER_T_ENTER + d_in + WORKER_PAUSE_S
    y_out, v_out, _ = _walk_segment(t, t_out, WORKER_Y_PAUSE, WORKER_Y0, WORKER_SPEED)
    leaving = t >= t_out
    y = jnp.where(leaving, y_out, y_in)
    v = jnp.where(leaving, v_out, v_in)
    return jnp.array([WORKER_X, y, 0.0, v])


def _safe_norm(v) -> Array:
    """``|v|`` with a finite gradient at zero (the paused worker has ``v = 0``)."""
    v = jnp.asarray(v, dtype=float)
    return jnp.sqrt(jnp.dot(v, v) + 1e-12)


def worker_yaw(v_xy) -> Array:
    """Facing direction from the walking velocity; +y (the entry direction) when standing.

    ``arctan2`` is only evaluated on a non-zero vector (the where/where pattern), so the
    barrier gradients stay finite while the worker stands still.
    """
    v = jnp.asarray(v_xy, dtype=float)
    moving = _safe_norm(v) > 1e-6
    vx = jnp.where(moving, v[0], 0.0)
    vy = jnp.where(moving, v[1], 1.0)
    return jnp.where(moving, jnp.arctan2(vy, vx), jnp.pi / 2)


def worker_quat(yaw) -> Array:
    """``(w, x, y, z)`` quaternion of a rotation ``yaw`` about z."""
    half = jnp.asarray(yaw, dtype=float) / 2.0
    return jnp.array([jnp.cos(half), 0.0, 0.0, jnp.sin(half)])


def worker_capsules(p_xy, yaw) -> Tuple[Array, Array, Array]:
    """World endpoints ``a, b`` ``(5, 3)`` and radii ``(5,)`` of the worker capsules."""
    p_xy = jnp.asarray(p_xy, dtype=float)
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    rot = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    base = jnp.array([p_xy[0], p_xy[1], 0.0])
    a = jnp.array([p0 for p0, _, _ in WORKER_CAPSULES]) @ rot.T + base
    b = jnp.array([p1 for _, p1, _ in WORKER_CAPSULES]) @ rot.T + base
    r = jnp.array([rr for _, _, rr in WORKER_CAPSULES])
    return a, b, r


def sphere_capsule_distance(c, r, a, b, rc) -> Array:
    """Surface distance between a sphere ``(c, r)`` and a capsule with axis ``a-b``, radius ``rc``."""
    c, a, b = (jnp.asarray(v, dtype=float) for v in (c, a, b))
    ab = b - a
    denom = jnp.maximum(jnp.dot(ab, ab), 1e-12)
    s = jnp.clip(jnp.dot(c - a, ab) / denom, 0.0, 1.0)
    closest = a + s * ab
    return jnp.linalg.norm(c - closest) - r - rc


# --------------------------------------------------------------------------- scene
def _collision_geoms(model: mujoco.MjModel, body: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return np.flatnonzero((model.geom_bodyid == bid) & (model.geom_group == 3))


def build_spec(*, offline: bool = False, showcase_floor: Optional[str] = None) -> mujoco.MjSpec:
    """The cell as an ``MjSpec``: iiwa14 on a pedestal, bin, conveyor, worker mannequin.

    ``showcase_floor`` (one of :data:`cbfkit.systems.mujoco.showcase.FLOOR_STYLES`) applies
    the showcase studio look for rendering; ``None`` keeps a plain physics scene.
    """
    from cbfkit.systems.mujoco import assets

    src = assets.iiwa14_model_dir(offline=offline) / "iiwa14.xml"
    spec = mujoco.MjSpec.from_file(str(src))
    spec.option.timestep = TIMESTEP
    world = spec.worldbody
    world.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.5, 0.5, 0.5, 1]
    )
    spec.body("base").pos = [0.0, 0.0, PEDESTAL_H]
    world.add_geom(
        name="pedestal",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[PEDESTAL_R, PEDESTAL_H / 2, 0],
        pos=[0.0, 0.0, PEDESTAL_H / 2],
        rgba=[0.30, 0.31, 0.33, 1],
    )
    world.add_geom(
        name="bin",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(BIN_HALF),
        pos=[BIN_XY[0], BIN_XY[1], BIN_HALF[2]],
        rgba=[0.20, 0.24, 0.30, 1],
    )
    for k, (dx, dy) in enumerate(((-0.08, -0.06), (0.07, 0.05), (0.0, -0.09))):
        world.add_geom(
            name=f"part{k}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.05, 0.035, 0.03],
            pos=[BIN_XY[0] + dx, BIN_XY[1] + dy, 2 * BIN_HALF[2] + 0.03],
            rgba=[0.85, 0.55, 0.15, 1],
            contype=0,
            conaffinity=0,
        )
    world.add_geom(
        name="conveyor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(CONVEYOR_HALF),
        pos=[CONVEYOR_XY[0], CONVEYOR_XY[1], CONVEYOR_HALF[2]],
        rgba=[0.36, 0.37, 0.40, 1],
    )
    world.add_geom(
        name="belt",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CONVEYOR_HALF[0] - 0.02, CONVEYOR_HALF[1] - 0.04, 0.005],
        pos=[CONVEYOR_XY[0], CONVEYOR_XY[1], 2 * CONVEYOR_HALF[2] + 0.005],
        rgba=[0.12, 0.12, 0.13, 1],
        contype=0,
        conaffinity=0,
    )
    link7 = spec.body("link7")
    link7.add_site(name="tool", pos=[0.0, 0.0, TOOL_OFFSET])
    link7.add_geom(
        name="tool_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[TOOL_R, 0, 0],
        pos=[0.0, 0.0, TOOL_OFFSET],
        group=3,
    )
    link7.add_geom(  # visual gripper block
        name="gripper",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.045, 0.03, 0.06],
        pos=[0.0, 0.0, TOOL_OFFSET - 0.02],
        rgba=[0.15, 0.15, 0.16, 1],
        contype=0,
        conaffinity=0,
        group=2,
    )
    worker = world.add_body(name="worker", mocap=True, pos=[WORKER_X, WORKER_Y0, 0.0])
    colours = {
        "legs": [0.22, 0.24, 0.30, 1],
        "torso": [1.0, 0.45, 0.05, 1],  # safety vest
        "head": [0.85, 0.70, 0.55, 1],
        "l_arm": [0.85, 0.70, 0.55, 1],
        "r_arm": [0.85, 0.70, 0.55, 1],
    }
    for name, (p0, p1, r) in zip(WORKER_CAPSULE_NAMES, WORKER_CAPSULES):
        if p0 == p1:
            worker.add_geom(
                name=f"worker_{name}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[r, 0, 0],
                pos=list(p0),
                rgba=colours[name],
            )
        else:
            worker.add_geom(
                name=f"worker_{name}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[*p0, *p1],
                size=[r, 0, 0],
                rgba=colours[name],
            )
    if showcase_floor is not None:
        from cbfkit.systems.mujoco import showcase

        showcase.apply_showcase_look(spec, showcase_floor)
    return spec


def load_cell(*, offline: bool = False) -> mujoco.MjModel:
    """Compiled physics model of the cell."""
    return build_spec(offline=offline).compile()


# --------------------------------------------------------------------------- plant
class CellPlant(MujocoPlant):
    """The cell plant: joint-velocity command plus the worker pose in one input vector.

    ``u = [qdot_cmd (7) | worker_xyz (3) | worker_quat (4)]``. The menagerie model's
    position servos (``kp`` 2000, ``kv`` 200) would only reach a fifth of a velocity
    commanded as ``qpos + qdot dt`` (the damping term wins), so the servo *target* is
    integrated in ``data.ctrl`` at the physics rate instead and clamped to
    ``qpos +/- CTRL_WINDUP`` -- a torque cap of about ``kp * CTRL_WINDUP`` when the arm is
    blocked, which is what keeps the unfiltered strike from winding up. The worker pose is
    written to the mocap body every substep.
    """

    def __init__(self, mj_model: mujoco.MjModel) -> None:
        super().__init__(mj_model, substeps=SUBSTEPS, nu=NU)
        rng = np.asarray(mj_model.jnt_range, dtype=float)
        self.q_min = jnp.asarray(rng[:, 0])
        self.q_max = jnp.asarray(rng[:, 1])
        self.tool_site = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "tool"))
        ids = []
        for body, k in ARM_SPHERE_GEOMS:
            geoms = _collision_geoms(mj_model, body)
            ids.append(int(geoms[k]))
        self.arm_geoms = tuple(ids)
        self.arm_radii = jnp.asarray(mj_model.geom_size[list(ids), 0], dtype=float)
        self.worker_body = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "worker"))
        self._template = mjx.make_data(mj_model)
        self.step = jax.jit(self.step)  # type: ignore[method-assign]
        self.from_state = jax.jit(self.from_state)  # type: ignore[method-assign]

    def from_state(self, x: Array) -> mjx.Data:
        """As the base class, plus the servo target seeded at the joint angles (no snap at t = 0)."""
        d = self._template.replace(qpos=x[: self.nq], qvel=x[self.nq : self.nq + self.nv])
        return mjx.forward(self.model, d.replace(ctrl=x[:N_ARM]))

    def make_data(self) -> mjx.Data:
        d = super().make_data()
        return d.replace(ctrl=d.qpos[:N_ARM])

    def step(self, data: mjx.Data, u: Array, model: Optional[mjx.Model] = None) -> mjx.Data:
        m = self.model if model is None else model
        qdot = u[:N_ARM]
        pos = u[N_ARM : N_ARM + 3][None, :]
        quat = u[N_ARM + 3 : NU][None, :]
        h = float(self.mj_model.opt.timestep)

        def _one(_, d):
            target = jnp.clip(d.ctrl[:N_ARM] + qdot * h, self.q_min, self.q_max)
            target = jnp.clip(target, d.qpos[:N_ARM] - CTRL_WINDUP, d.qpos[:N_ARM] + CTRL_WINDUP)
            return mjx.step(m, d.replace(ctrl=target, mocap_pos=pos, mocap_quat=quat))

        return jax.lax.fori_loop(0, self.substeps, _one, data)

    def kinematics(self, q: Array) -> mjx.Data:
        """Forward kinematics only (no dynamics) at joint configuration ``q``."""
        return mjx.kinematics(self.model, self._template.replace(qpos=q))


def make_plant(*, offline: bool = False) -> CellPlant:
    """Build the cell plant (``dt = 0.02``, ``nu = 14``, ``state_dim = 17``)."""
    return CellPlant(load_cell(offline=offline))


# --------------------------------------------------------------------------- barriers
def arm_sphere_positions(plant: CellPlant, q) -> Tuple[Array, Array]:
    """World centres ``(8, 3)`` and radii ``(8,)`` of the representative arm spheres."""
    d = plant.kinematics(jnp.asarray(q, dtype=float))
    return d.geom_xpos[jnp.array(plant.arm_geoms)], plant.arm_radii


def _pair_h(plant: CellPlant, q, worker4, i: int, j: int) -> Array:
    centres, radii = arm_sphere_positions(plant, q)
    a, b, r = worker_capsules(worker4[:2], worker_yaw(worker4[2:]))
    margin = D0 + TAU * _safe_norm(worker4[2:])
    return sphere_capsule_distance(centres[i], radii[i], a[j], b[j], r[j]) - margin


def barrier_values(plant: CellPlant, q, worker4) -> Array:
    """All ``h_ij`` as a flat ``(40,)`` vector, sphere-major, in :data:`H_NAMES` order."""
    q = jnp.asarray(q, dtype=float)
    worker4 = jnp.asarray(worker4, dtype=float)
    centres, radii = arm_sphere_positions(plant, q)
    a, b, r = worker_capsules(worker4[:2], worker_yaw(worker4[2:]))
    margin = D0 + TAU * _safe_norm(worker4[2:])

    def one_sphere(c, rc):
        return jax.vmap(lambda aa, bb, rr: sphere_capsule_distance(c, rc, aa, bb, rr))(a, b, r)

    return (jax.vmap(one_sphere)(centres, radii) - margin).reshape(-1)


def arm_worker_barriers(
    plant: CellPlant, *, class_k_gain: float = CLASS_K
) -> CertificateCollection:
    """The 40 pairwise sphere-capsule barriers on the augmented state ``[x | worker4]``."""
    n = plant.state_dim + 4

    def factory(i: int, j: int):
        def h(x):
            return _pair_h(plant, x[:N_ARM], x[plant.state_dim : plant.state_dim + 4], i, j)

        return h

    package = certificate_package(factory, n=n, input_style="state")
    conditions = zeroing_barriers.linear_class_k(class_k_gain)
    return concatenate_certificates(
        *[
            package(certificate_conditions=conditions, i=i, j=j)
            for i in range(N_SPHERES)
            for j in range(N_CAPSULES)
        ]
    )


def joint_limit_barriers(plant: CellPlant, *, margin: float = JOINT_LIMIT_MARGIN, class_k_gain: float = CLASS_K) -> CertificateCollection:
    """Fourteen relative-degree-1 barriers keeping each joint ``margin`` inside its range."""
    n = plant.state_dim + 4
    q_min = np.asarray(plant.q_min) + margin
    q_max = np.asarray(plant.q_max) - margin

    def factory(i: int, upper: bool):
        def h(x):
            return (q_max[i] - x[i]) if upper else (x[i] - q_min[i])

        return h

    package = certificate_package(factory, n=n, input_style="state")
    conditions = zeroing_barriers.linear_class_k(class_k_gain)
    return concatenate_certificates(
        *[
            package(certificate_conditions=conditions, i=i, upper=upper)
            for i in range(N_ARM)
            for upper in (False, True)
        ]
    )


def embedded_arm_worker(state_dim: int) -> DynamicsCallable:
    """Reduced model on ``[x | px, py, vx, vy]``: ``d/dt qpos = u`` and a constant-velocity worker."""
    n = state_dim + 4
    g = jnp.zeros((n, N_ARM)).at[:N_ARM, :].set(jnp.eye(N_ARM))

    def dynamics(xa: Array):
        f = (
            jnp.zeros(n)
            .at[state_dim]
            .set(xa[state_dim + 2])
            .at[state_dim + 1]
            .set(xa[state_dim + 3])
        )
        return f, g

    return dynamics


# --------------------------------------------------------------------------- nominal task
def tool_pose(plant: CellPlant, q) -> Tuple[Array, Array]:
    """Tool site position ``(3,)`` and its z-axis in the world ``(3,)``."""
    d = plant.kinematics(jnp.asarray(q, dtype=float))
    return d.site_xpos[plant.tool_site], d.site_xmat[plant.tool_site].reshape(3, 3)[:, 2]


def _segments() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The pick-place cycle as ``(t_start, duration, p0, p1)`` arrays; dwell = ``p0 == p1``."""
    bin_above = np.array([BIN_XY[0], BIN_XY[1], Z_ABOVE_BIN])
    bin_pt = np.array([BIN_XY[0], BIN_XY[1], Z_BIN])
    conv_above = np.array([CONVEYOR_XY[0], CONVEYOR_XY[1], Z_ABOVE_CONV])
    conv_pt = np.array([CONVEYOR_XY[0], CONVEYOR_XY[1], Z_CONV])
    legs = [
        (bin_above, conv_above),
        (conv_above, conv_pt),
        (conv_pt, conv_pt),
        (conv_pt, conv_above),
        (conv_above, bin_above),
        (bin_above, bin_pt),
        (bin_pt, bin_pt),
        (bin_pt, bin_above),
    ]
    starts, durs, p0s, p1s = [], [], [], []
    t = 0.0
    for p0, p1 in legs:
        dist = float(np.linalg.norm(p1 - p0))
        if dist < 1e-9:
            dur = DWELL_S
        elif dist < V_TOOL**2 / A_TOOL:  # triangular profile
            dur = 2.0 * float(np.sqrt(dist / A_TOOL))
        else:
            dur = dist / V_TOOL + V_TOOL / A_TOOL
        starts.append(t)
        durs.append(dur)
        p0s.append(p0)
        p1s.append(p1)
        t += dur
    return np.array(starts), np.array(durs), np.array(p0s), np.array(p1s)


_SEG_START, _SEG_DUR, _SEG_P0, _SEG_P1 = _segments()
CYCLE_S = float(_SEG_START[-1] + _SEG_DUR[-1])


def _profile(tau, dist, dur):
    """Distance travelled and speed along a trapezoid/triangle of length ``dist`` and time ``dur``."""
    # Peak speed and acceleration time from the closed forms used in _segments.
    triangular = dist < V_TOOL**2 / A_TOOL
    t_acc = jnp.where(triangular, dur / 2.0, V_TOOL / A_TOOL)
    v_peak = jnp.where(triangular, A_TOOL * t_acc, V_TOOL)
    t_dec = dur - t_acc
    tau = jnp.clip(tau, 0.0, dur)
    s_acc = 0.5 * A_TOOL * tau**2
    v_acc = A_TOOL * tau
    s_cruise = 0.5 * A_TOOL * t_acc**2 + v_peak * (tau - t_acc)
    s_dec = dist - 0.5 * A_TOOL * (dur - tau) ** 2
    v_dec = A_TOOL * (dur - tau)
    s = jnp.where(tau < t_acc, s_acc, jnp.where(tau < t_dec, s_cruise, s_dec))
    v = jnp.where(tau < t_acc, v_acc, jnp.where(tau < t_dec, v_peak, v_dec))
    dwell = dist < 1e-9
    return jnp.where(dwell, 0.0, s), jnp.where(dwell, 0.0, v)


def pick_place_reference(t) -> Tuple[Array, Array]:
    """Tool reference position and velocity on the periodic bin-conveyor cycle."""
    tau = jnp.mod(jnp.asarray(t, dtype=float), CYCLE_S)
    k = jnp.clip(
        jnp.searchsorted(jnp.asarray(_SEG_START), tau, side="right") - 1, 0, len(_SEG_START) - 1
    )
    p0 = jnp.asarray(_SEG_P0)[k]
    p1 = jnp.asarray(_SEG_P1)[k]
    dist = jnp.linalg.norm(p1 - p0)
    direction = (p1 - p0) / jnp.maximum(dist, 1e-9)
    s, v = _profile(tau - jnp.asarray(_SEG_START)[k], dist, jnp.asarray(_SEG_DUR)[k])
    return p0 + s * direction, v * direction


def osc_velocity(plant: CellPlant, q, phase) -> Array:
    """Resolved-rate joint velocity toward :func:`pick_place_reference` at cycle time ``phase``.

    Task: tool position plus tool axis pointing down. The position command
    ``KP_POS * e + v_des`` is saturated at ``V_TOOL`` so a distant reference asks for the
    cycle speed, not more; damped pseudo-inverse with a null-space pull toward
    :data:`Q_HOME`; clipped to the datasheet joint speeds.
    """
    q_home = jnp.asarray(Q_HOME)
    a_des = jnp.array([0.0, 0.0, -1.0])
    lam2 = 1e-3

    def task(qq):
        p, a = tool_pose(plant, qq)
        return jnp.concatenate([p, a])

    y = task(q)
    jac = jax.jacfwd(task)(q)
    p_des, v_des = pick_place_reference(phase)
    v_cmd = KP_POS * (p_des - y[:3]) + v_des
    speed = _safe_norm(v_cmd)
    v_cmd = jnp.where(speed > V_TOOL, v_cmd * (V_TOOL / speed), v_cmd)
    err = jnp.concatenate([v_cmd, KP_ORI * (a_des - y[3:])])
    jjt = jac @ jac.T + lam2 * jnp.eye(6)
    j_pinv = jac.T @ jnp.linalg.solve(jjt, jnp.eye(6))
    qd = j_pinv @ err + (jnp.eye(N_ARM) - j_pinv @ jac) @ (KP_NULL * (q_home - q))
    lim = jnp.asarray(QD_LIMITS)
    return jnp.clip(qd, -lim, lim)


def phase_rate(plant: CellPlant, q, phase) -> Array:
    """How fast the cycle clock runs (0..1) given the tool's distance to its reference."""
    p_des, _ = pick_place_reference(phase)
    e = _safe_norm(p_des - tool_pose(plant, q)[0])
    return jnp.clip(1.0 - (e - CATCHUP_R) / CATCHUP_BAND, 0.0, 1.0)


def osc_nominal(plant: CellPlant):
    """Stateless variant of :func:`osc_velocity` on the wall clock, for tests and the simulator.

    Signature matches ``simulator.execute(nominal_controller=...)``:
    ``(t, x, key, ref) -> (qdot, ControllerData)``. :func:`safe_arm_controller` does not use
    this value -- it recomputes the nominal from the cycle phase it carries -- but
    ``execute`` needs a nominal and this keeps ``u_nom`` meaningful in the log.
    """

    def nominal(t, x, key=None, ref=None):
        return osc_velocity(plant, x[:N_ARM], t), ControllerData()

    return nominal


# --------------------------------------------------------------------------- safety layer
def safe_arm_controller(cbf_qp: ControllerCallable, plant: CellPlant) -> ControllerCallable:
    """Filter the phase-held OSC nominal through ``cbf_qp`` on ``[x | worker]`` and build ``u``.

    The cycle phase is carried in ``sub_data["_phase"]`` and advances at
    :func:`phase_rate` (1 while the tool tracks, 0 when it is held away from its
    reference), so the nominal resumes the cycle where it left off. The ``qdot_nom`` the
    simulator passes in is ignored (see :func:`osc_nominal`).

    Logged in ``sub_data``: ``qdot_nom``, ``qdot_safe`` (7 each), ``worker`` (4), ``tool``
    (3), ``phase`` (1) and ``h`` (35, :data:`H_NAMES` order) plus whatever the QP logs.
    """

    def controller(t, x, _qdot_nom_unused, key, data):
        prev = data.sub_data if data.sub_data is not None else {}
        phase = prev.get("_phase")
        if phase is None:
            phase = jnp.asarray(0.0)
        q = x[:N_ARM]
        w = worker_state(t)
        qdot_nom = osc_velocity(plant, q, phase)
        xa = jnp.concatenate([x, w])
        qdot_safe, d1 = cbf_qp(t, xa, qdot_nom, key, data)
        qdot_safe = jnp.asarray(qdot_safe)[:N_ARM]
        yaw = worker_yaw(w[2:])
        u = jnp.concatenate([qdot_safe, jnp.array([w[0], w[1], 0.0]), worker_quat(yaw)])
        sub = dict(d1.sub_data) if d1.sub_data is not None else {}
        for k, val in prev.items():
            if k.startswith("_") and k not in sub:
                sub[k] = val
        sub["_phase"] = phase + plant.dt * phase_rate(plant, q, phase)
        sub["phase"] = phase
        sub["qdot_nom"] = qdot_nom
        sub["qdot_safe"] = qdot_safe
        sub["worker"] = w
        sub["tool"] = tool_pose(plant, q)[0]
        sub["h"] = barrier_values(plant, q, w)
        return u, d1._replace(sub_data=sub, u=u, u_nom=qdot_nom)

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
    return controller


def build(*, unfiltered: bool = False, offline: bool = False):
    """``(plant, x0, controller, nominal)`` for ``simulator.execute(plant=...)``.

    ``unfiltered`` swaps the barrier collection for the empty one, so the QP is the
    projection onto the velocity box and ``qdot_safe == clip(qdot_nom)``.
    """
    plant = make_plant(offline=offline)
    barriers = (
        concatenate_certificates()
        if unfiltered
        else joint_limit_barriers(plant) + arm_worker_barriers(plant)
    )
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.asarray(QD_LIMITS),
        dynamics_func=embedded_arm_worker(plant.state_dim),
        barriers=barriers,
        relaxable_cbf=not unfiltered,
        slack_penalty_cbf=SLACK_PENALTY,
        solver=get_solver(QP_SOLVER, tol=QP_TOL, max_iter=QP_MAX_ITER),
    )
    x0 = plant.to_state(plant._template.replace(qpos=jnp.asarray(Q_HOME)))
    return plant, x0, safe_arm_controller(cbf_qp, plant), osc_nominal(plant)


def _solve_home(plant: CellPlant, iters: int = 200) -> np.ndarray:
    """Gauss-Newton IK for :data:`Q_HOME` (kept for re-deriving the constant, not used at runtime)."""
    target = jnp.array([BIN_XY[0], BIN_XY[1], Z_ABOVE_BIN, 0.0, 0.0, -1.0])
    q = jnp.array([-0.8, 0.7, 0.0, -1.4, 0.0, 1.0, -0.8])

    def task(qq):
        p, a = tool_pose(plant, qq)
        return jnp.concatenate([p, a])

    for _ in range(iters):
        jac = jax.jacfwd(task)(q)
        err = target - task(q)
        q = q + jnp.linalg.pinv(jac, rcond=1e-3) @ err * 0.5
        q = jnp.clip(q, plant.q_min, plant.q_max)
    return np.asarray(q)
