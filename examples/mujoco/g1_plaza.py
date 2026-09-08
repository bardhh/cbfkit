"""Unitree G1 crosses a plaza among reactive pedestrians: waypoint route, static pillars and
social-force pedestrians under high-order CBFs on the CoM.

The richer sibling of ``g1_navigate.py``. Everything runs inside one CBFKit
``execute(plant=...)`` call and exercises the whole pipeline:

    planner:    waypoint_route            -> current waypoint (PlannerData.x_traj)
    nominal:    v_nom = k (wp - com)      (2-D, saturated at V_MAX)
    safety:     robust CBF-QP on the command-side double integrator  d/dt com = v, d/dt v = a
                  * 2 static pillars     -> com_obstacle_hocbfs   (relative degree 2)
                  * 3 pedestrians        -> com_agent_hocbfs      (tracked agents in the state:
                                            d/dt p_i = v_i, a constant-velocity prediction from
                                            the pedestrian's *current* velocity)
    locomotion: Unitree's pretrained walking policy tracks the integrated velocity command,
                turning to face it (heading follower)
    plant:      MJX G1 (12-DoF legs, PD at 500 Hz)

The pedestrians are *reactive*: ``SocialForceCrowd`` steps them inside the safety wrapper
with the social-force model (goal attraction; repulsion from the robot, from each other and
from the pillars), so they yield to the robot while the robot yields to them. They are
kinematic -- not MuJoCo bodies -- and drawn into the GIF/viewer as moving capsules. The CBF
sees their tracked position and velocity; their *accelerations* (the reactions) are the
unmodelled part, reported below as a prediction-error statistic. Encounters are built to
collide: P1 walks head-on into the robot mid leg 2, P3 cuts across right in front of it on
leg 1, P2 ambles ahead of it along leg 3 and must be overtaken; both pillars sit inside the
keep-out of the straight nominal legs.

Barrier shape (``--barrier``): ``distance`` (default) is ``h = |com - p| / r - 1`` with a
unit-norm gradient, so the robust margin ``|dh/dx| * delta`` and the HOCBF approach limit do
not grow with distance; ``ellipsoid`` is the stock quadratic barrier, which keeps a wide
berth far away (compare the two with ``--gif``).

Safety measure: ``h_min`` over all obstacles (>= 0 means the CoM never entered a keep-out
disc). The robust bound is the measured tracking error ``||v_com - v_safe||`` of the gait *in
this scenario* (``--robust``; 0 = vanilla); the default 0.31 sits above the max observed under
it (0.298). Measured (CPU, 45 s horizon, seed 0; "closest" = CoM-pedestrian distance, keep-out
0.65 m; pedestrian reactions = min speed / max sidestep from their intended line):

    barrier    CBF      bound  h_min   closest  W1/W2/W3 [s]        tracking mean/p95/max   pedestrians P1 / P2 / P3
    distance   vanilla  --     -0.048  0.88 m   8.5 / 20.3 / 27.2   0.078 / 0.149 / 0.279   stops,1.6 m / 0.13,0.2 m / stops,0.7 m
    distance   robust   0.31*  +0.588  1.18 m   10.1 / 31.5 / 39.4  0.079 / 0.157 / 0.298   stops,0.9 m / 0.15,0 m / stops,0.7 m
    ellipsoid  vanilla  --     -0.103  0.92 m   8.7 / 21.1 / 27.9   0.076 / 0.145 / 0.210   stops,1.6 m / 0.13,0.3 m / stops,0.7 m
    ellipsoid  robust   0.31   QP infeasible at t = 6.0 s: the quadratic barrier's gradient grows with distance, so the
                               margins of the *far* pedestrians alone exceed the acceleration budget (that is why
                               ``distance`` is the default; ``g1_navigate`` gets away with it because it has one barrier)

Read the vanilla row as the point of the exercise: the gait tracks the certified command only
approximately, so the vanilla CBF lets the CoM graze pillar 2 (h_min < 0) while the robust CBF
holds every barrier with margin and the robot visibly gives way -- it swings around the far side
of pillar 2 for the oncoming P1, brakes for P3 cutting across, and overtakes P2 near the goal --
at the price of a slower crossing (39 s vs 27 s). The pedestrians give way too (P1 and P3 stop
and step aside). Solver: the in-repo PDIPM (``get_solver("fast")``); with both robust margins
active between pillar 2 and P1 the feasible set is a thin wedge and jaxopt's OSQP hits its
iteration limit on this 2-variable QP (reproducible at t = 16.3 s of the robust run).

    python examples/mujoco/g1_plaza.py [--robust B] [--barrier distance|ellipsoid] [--duration T] [--gif] [--view]
"""

import argparse
import os
import sys
import time

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax
import jax.numpy as jnp
import numpy as np

import cbfkit.simulation.simulator as sim
from cbfkit.certificates import concatenate_certificates
from cbfkit.controllers.cbf_clf import robust_cbf_clf_qp_controller, vanilla_cbf_clf_qp_controller
from cbfkit.optimization.quadratic_program.solver_registry import get_solver
from cbfkit.planners import waypoint_route
from cbfkit.systems.mujoco.crowd import SocialForceCrowd
from cbfkit.systems.mujoco.reduced_order import (
    com_agent_hocbfs,
    com_obstacle_hocbfs,
    embedded_double_integrator,
    safe_locomotion_controller_di,
)
from cbfkit.systems.mujoco.unitree_policy import (
    UnitreeG1WalkPolicy,
    make_g1_12dof_plant,
    x0_standing,
)
from cbfkit.systems.mujoco.viewer_utils import (
    add_marker,
    relaunch_under_mjpython_if_needed,
    render_gif,
    replay_in_viewer,
)
from cbfkit.utils.user_types import ControllerData, PlannerData

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# --------------------------------------------------------------------------- scenario
WAYPOINTS = jnp.array([[3.0, 1.2], [6.0, -1.0], [9.0, 0.3]])  # the route, from (0, 0)
WAYPOINT_RADIUS = 0.35  # switch to the next waypoint inside this
GOAL_RADIUS = 0.3  # completion at the last one
PILLARS = jnp.array([[1.6, 0.9], [4.6, -0.2]])  # both within the keep-out of the straight legs
PILLAR_RADIUS = 0.30
# Pedestrians: (start, goal, desired speed). Goals lie beyond the plaza so they walk through.
PED_STARTS = jnp.array([[8.5, -2.85], [2.8, -2.4], [0.8, 3.0]])
PED_GOALS = jnp.array([[-1.5, 4.5], [12.0, -0.2], [2.6, -6.0]])
PED_SPEEDS = jnp.array([0.40, 0.15, 0.60])
PED_NOTES = [
    "head-on along leg 2 (meets the robot just past W1)",
    "ambles ahead along leg 3 (robot must overtake)",
    "cuts across leg 1 right in front of the robot",
]
PED_RADIUS = 0.30
PED_REPULSION_RANGE = 0.4  # social-force range: pedestrians start yielding ~1.05 m from the robot
ROBOT_RADIUS = 0.35  # planar inflation so "CoM outside the disc" => "body clear"
V_MAX = 0.5
A_MAX = 1.0  # double-integrator model: acceleration limit [m/s^2]
BARRIER_SHAPE = "distance"
DEFAULT_ROBUST_BOUND = 0.31  # measured in this scenario, see the docstring
DEFAULT_DURATION = 45.0
_TAG = ["run"]  # results filename tag, set per run

R_PILLAR = PILLAR_RADIUS + ROBOT_RADIUS
R_PED = PED_RADIUS + ROBOT_RADIUS
N_PED = int(PED_STARTS.shape[0])
OBSTACLE_NAMES = [f"pillar {i+1}" for i in range(len(PILLARS))] + [
    f"pedestrian {i+1}" for i in range(N_PED)
]


def build(seed: int = 0, robust_bound: float = 0.0, barrier_shape: str = BARRIER_SHAPE):
    """Plant, policy, planner, nominal and safety controller for the plaza scenario."""
    plant = make_g1_12dof_plant()  # 12-DoF legs, PD at 500 Hz, dt 0.02
    loco = UnitreeG1WalkPolicy().as_controller()
    x0 = x0_standing(plant)
    pelvis_body = int(plant.mj_model.body("pelvis").id)
    ci = plant.com_indices

    crowd = SocialForceCrowd(
        PED_STARTS,
        PED_GOALS,
        PED_SPEEDS,
        obstacles=PILLARS,
        ped_radius=PED_RADIUS,
        agent_radius=ROBOT_RADIUS,
        repulsion_range=PED_REPULSION_RANGE,
    )
    dyn = embedded_double_integrator(plant.state_dim, ci, n_agents=N_PED)
    barriers = concatenate_certificates(
        com_obstacle_hocbfs(
            plant,
            PILLARS,
            [(R_PILLAR, R_PILLAR)] * len(PILLARS),
            class_k_gain=1.0,
            shape=barrier_shape,
            n_agents=N_PED,
        ),
        com_agent_hocbfs(
            plant, N_PED, [(R_PED, R_PED)] * N_PED, class_k_gain=1.0, shape=barrier_shape
        ),
    )
    limits = jnp.array([A_MAX, A_MAX])
    # The in-repo PDIPM: when the robot is squeezed between a pillar and an oncoming pedestrian
    # with both robust margins active, the feasible set is a thin wedge and jaxopt's OSQP (the
    # default) hits its iteration limit on this 2-variable QP; the PDIPM solves it in ~16 steps.
    solver = get_solver("fast")
    if robust_bound > 0.0:
        cbf_qp = robust_cbf_clf_qp_controller(
            control_limits=limits,
            dynamics_func=dyn,
            barriers=barriers,
            disturbance_norm=2,
            disturbance_norm_bound=float(robust_bound),
            solver=solver,
        )
    else:
        cbf_qp = vanilla_cbf_clf_qp_controller(
            control_limits=limits, dynamics_func=dyn, barriers=barriers, solver=solver
        )
    safe = safe_locomotion_controller_di(cbf_qp, loco, plant, plant.dt, v_max=V_MAX, agents=crowd)

    def controller(t, x, v_nom, key, data):  # + completion at the last waypoint (early stop)
        u, d = safe(t, x, v_nom, key, data)
        reached = jnp.linalg.norm(x[ci[0] : ci[0] + 2] - WAYPOINTS[-1]) < GOAL_RADIUS
        return u, d._replace(complete=d.complete | reached)

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]

    def nominal(t, x, key, ref):  # proportional to the current waypoint on the CoM, saturated
        com = x[ci[0] : ci[0] + 2]
        v = 1.0 * (jnp.asarray(ref)[:2] - com)
        speed = jnp.linalg.norm(v)
        v = jnp.where(speed > V_MAX, v * (V_MAX / (speed + 1e-9)), v)
        return v, ControllerData()

    planner = waypoint_route(WAYPOINTS, WAYPOINT_RADIUS, position_indices=ci)
    return plant, x0, pelvis_body, planner, nominal, controller


def barrier_values(com: np.ndarray, agents: np.ndarray, shape: str = BARRIER_SHAPE) -> np.ndarray:
    """``h`` per obstacle over the run, shape ``(T, n_pillars + n_pedestrians)``.

    ``agents`` is the logged ``(T, N, 4)`` pedestrian state (as the QP saw it).
    """

    def h(d2, r):
        q = d2 / r**2
        return np.sqrt(q) - 1 if shape == "distance" else q - 1

    hs = []
    for i in range(len(PILLARS)):
        hs.append(h(((com - np.asarray(PILLARS[i])) ** 2).sum(1), R_PILLAR))
    for i in range(N_PED):
        hs.append(h(((com - agents[:, i, :2]) ** 2).sum(1), R_PED))
    return np.stack(hs, axis=1)


def waypoint_arrivals(com: np.ndarray) -> list:
    """First step index at which each waypoint (in order) is reached, or None."""
    out, start = [], 0
    for wp in np.asarray(WAYPOINTS):
        d = np.linalg.norm(com[start:] - wp, axis=1)
        hit = np.flatnonzero(d < WAYPOINT_RADIUS)
        if hit.size == 0:
            break
        start += int(hit[0])
        out.append(start)
    while len(out) < len(WAYPOINTS):
        out.append(None)
    return out


def main(
    duration=DEFAULT_DURATION,
    seed=0,
    gif=False,
    view=False,
    robust_bound=None,
    barrier_shape=BARRIER_SHAPE,
):
    if robust_bound is None:  # measured in this scenario, see the module docstring
        robust_bound = DEFAULT_ROBUST_BOUND
    plant, x0, pelvis_body, planner, nominal, controller = build(seed, robust_bound, barrier_shape)
    steps = 5 if TEST_MODE else int(round(duration / plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        plant=plant,
        planner=planner,
        planner_data=PlannerData.from_constant(WAYPOINTS[0]),
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    wall = time.time() - t0
    states = np.asarray(res["states"])
    ci = plant.com_indices
    com = states[:, ci[0] : ci[0] + 2]
    t = np.arange(len(com)) * plant.dt
    cd = res.controller_data
    agents = np.asarray(cd["sub_data_agents"])  # (T, N, 4) as seen by the QP each step
    H = barrier_values(com, agents, barrier_shape)
    v_nom = np.asarray(cd.get("sub_data_v_nom")) if "sub_data_v_nom" in cd else None
    v_safe = np.asarray(cd.get("sub_data_v_safe")) if "sub_data_v_safe" in cd else None
    hh = states[:, 2]  # pelvis height
    q = states[:, 3:7]  # pelvis quaternion (w, x, y, z): body z-axis . world z
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    arrivals = waypoint_arrivals(com)
    n_live = arrivals[-1] if arrivals[-1] is not None else len(com)

    print(f"{steps} steps in {wall:.1f}s")
    print(f"h(x) min over run: {H.min():.3f}   (>= 0 means the CoM never entered a keep-out disc)")
    for name, hcol in zip(OBSTACLE_NAMES, H.T):
        k = int(np.argmin(hcol))
        print(f"  {name:13s} h_min {hcol[k]:+.3f} at t={t[k]:.1f}s")
    ped_d = np.linalg.norm(com[:, None, :] - agents[:, :, :2], axis=2)  # (T, N)
    print(f"closest CoM-pedestrian distance: {ped_d.min():.2f} m (keep-out {R_PED:.2f})")
    print(
        "waypoints reached at: "
        + ", ".join("--" if k is None else f"{k*plant.dt:.1f}s" for k in arrivals)
        + ("" if arrivals[-1] is not None else "  (route NOT completed)")
    )
    print(f"pelvis height min {hh.min():.2f}, upright min {up.min():.2f}")
    # Pedestrian reactions (the unmodelled part of the agent model: the CBF predicts them at
    # constant velocity, so their acceleration is the prediction error).
    vel = agents[:n_live, :, 2:]
    acc = np.gradient(vel, plant.dt, axis=0)
    for i in range(N_PED):
        sp = np.linalg.norm(vel[:, i], axis=1)
        p0, g = np.asarray(PED_STARTS[i]), np.asarray(PED_GOALS[i])
        u = (g - p0) / np.linalg.norm(g - p0)
        off = agents[:n_live, i, :2] - p0
        lateral = np.abs(off[:, 0] * u[1] - off[:, 1] * u[0])  # deviation from the straight line
        print(
            f"  pedestrian {i+1}: {PED_NOTES[i]}; closest {ped_d[:n_live, i].min():.2f} m at "
            f"t={t[np.argmin(ped_d[:n_live, i])]:.1f}s, speed min {sp.min():.2f} of {float(PED_SPEEDS[i]):.2f}, "
            f"max sidestep {lateral.max():.2f} m, max |acc| {np.linalg.norm(acc[:, i], axis=1).max():.2f} m/s^2"
        )
    if v_nom is not None and v_safe is not None:
        print(
            f"CBF active (|v_safe - v_nom| > 1e-3) on {np.mean(np.linalg.norm(v_safe - v_nom, axis=1) > 1e-3)*100:.0f}% of steps"
        )
        v_com = np.gradient(com[:n_live], plant.dt, axis=0)
        err = np.linalg.norm(v_com - v_safe[:n_live], axis=1)
        print(
            f"tracking error ||v_com - v_safe||: mean {err.mean():.3f}, p95 {np.percentile(err, 95):.3f}, "
            f"max {err.max():.3f} m/s  (use as --robust bound)"
        )
    if TEST_MODE:
        return float(H.min())
    _TAG[0] = f"{barrier_shape}_{'robust' if robust_bound > 0 else 'vanilla'}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # The simulation latches after completion (state frozen); plot/render only the live part.
    n = min(len(com), n_live + int(1.0 / plant.dt))
    _plot(com[:n], t[:n], H[:n], agents[:n], v_nom[:n], v_safe[:n])
    markers = _make_markers(agents)
    if gif:
        path = os.path.join(RESULTS_DIR, f"g1_plaza_{_TAG[0]}.gif")
        render_gif(plant, states[:n], path, track_body=pelvis_body, markers=markers, distance=5.5)
    if view:
        replay_in_viewer(plant, states[:n], markers=markers)
    return float(H.min())


def _plot(com, t, H, agents, v_nom, v_safe):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [2, 1.2, 1]})
    ax = axes[0]
    for p in np.asarray(PILLARS):
        ax.add_patch(plt.Circle(p, PILLAR_RADIUS, color="crimson", alpha=0.7))
        ax.add_patch(plt.Circle(p, R_PILLAR, color="crimson", fill=False, ls="--", lw=0.8))
    for i in range(N_PED):
        p0, g = np.asarray(PED_STARTS[i]), np.asarray(PED_GOALS[i])
        ax.plot([p0[0], g[0]], [p0[1], g[1]], ":", color="tab:blue", lw=0.8)  # intent
        traj = agents[:, i, :2]
        ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:blue", lw=1.2)  # actual
        ax.plot(*p0, "o", color="tab:blue", ms=5)
        ax.annotate(f"P{i+1}", p0, textcoords="offset points", xytext=(4, 4), color="tab:blue")
        for k in range(0, len(t), max(1, len(t) // 8)):  # snapshots along the walk
            ax.add_patch(plt.Circle(traj[k], PED_RADIUS, color="tab:blue", alpha=0.12))
    sc = ax.scatter(com[:, 0], com[:, 1], c=t, cmap="viridis", s=6, label="CoM (colour = t)")
    for i, wp in enumerate(np.asarray(WAYPOINTS)):
        ax.plot(*wp, "g*", ms=13)
        ax.annotate(f"W{i+1}", wp, textcoords="offset points", xytext=(5, 5), color="green")
    ax.plot(0, 0, "ks", ms=6, label="start")
    ax.set_aspect("equal")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-5, 4.5)
    ax.legend(loc="lower right")
    ax.set_title("plaza: CoM path, pillars (red), pedestrians (blue: intent dotted, actual solid)")
    fig.colorbar(sc, ax=ax, label="t [s]", shrink=0.7)
    for name, hcol in zip(OBSTACLE_NAMES, H.T):
        axes[1].plot(t, hcol, lw=1.5, label=name)
    axes[1].axhline(0, color="k", ls=":")
    axes[1].set_ylim(-0.3, 4)
    axes[1].set_title("barriers h_i (>= 0 safe)")
    axes[1].set_xlabel("t [s]")
    axes[1].legend(fontsize=8)
    if v_nom is not None:
        axes[2].plot(t, np.linalg.norm(v_nom, axis=1), "--", label="|v_nom|")
        axes[2].plot(t, np.linalg.norm(v_safe, axis=1), label="|v_safe|")
        v_com = np.gradient(com, t[1] - t[0], axis=0)
        axes[2].plot(
            t, np.linalg.norm(v_com, axis=1), lw=0.6, alpha=0.6, label="|v_com| (measured)"
        )
        axes[2].legend(fontsize=8)
        axes[2].set_title("CoM speed command vs measured")
        axes[2].set_xlabel("t [s]")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"g1_plaza_{_TAG[0]}.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def _make_markers(agents):
    """Markers callback: pillars, pedestrians at frame k (from the log), rings, waypoints, goal."""
    import mujoco

    cyl, sph, cap = (
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
    )
    pillars = np.asarray(PILLARS)
    wps = np.asarray(WAYPOINTS)
    last = min(len(agents) - 1, 10**9)

    def markers(scn, k, t):
        for p in pillars:
            add_marker(scn, cyl, [PILLAR_RADIUS, 0.6, 0], [*p, 0.6], [0.85, 0.15, 0.2, 1.0])
            add_marker(scn, cyl, [R_PILLAR, 0.005, 0], [*p, 0.005], [0.85, 0.15, 0.2, 0.25])
        for p in agents[min(k, last), :, :2]:
            add_marker(scn, cap, [PED_RADIUS * 0.6, 0.55, 0], [*p, 0.85], [0.2, 0.4, 0.9, 1.0])
            add_marker(scn, sph, [PED_RADIUS * 0.5, 0, 0], [*p, 1.55], [0.2, 0.4, 0.9, 1.0])
            add_marker(scn, cyl, [R_PED, 0.005, 0], [*p, 0.005], [0.2, 0.4, 0.9, 0.25])
        for i, wp in enumerate(wps):
            is_goal = i == len(wps) - 1
            r = GOAL_RADIUS if is_goal else 0.12
            add_marker(scn, sph, [r, 0, 0], [*wp, r], [0.1, 0.8, 0.2, 0.6 if is_goal else 0.9])

    return markers


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    p.add_argument(
        "--robust",
        type=float,
        default=None,
        help="robust CBF with this ||v_com - v_safe|| bound in m/s (default: measured, see docstring); 0 = vanilla",
    )
    p.add_argument(
        "--barrier",
        default=BARRIER_SHAPE,
        choices=["distance", "ellipsoid"],
        help="keep-out barrier shape: distance (|c-p|/r - 1, default) or the stock quadratic ellipsoid",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    if a.view:
        relaunch_under_mjpython_if_needed()
    main(a.duration, a.seed, a.gif, a.view, a.robust, a.barrier)
