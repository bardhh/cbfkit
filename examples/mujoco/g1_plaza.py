"""Unitree G1 crosses a plaza: waypoint route, static pillars and walking pedestrians under
time-varying high-order CBFs on the CoM.

The richer sibling of ``g1_navigate.py``. Everything runs inside one CBFKit
``execute(plant=...)`` call and exercises the whole pipeline:

    planner:    waypoint_route          -> current waypoint (PlannerData.x_traj)
    nominal:    v_nom = k (wp - com)    (2-D, saturated at V_MAX)
    safety:     robust CBF-QP on the command-side double integrator  d/dt com = v, d/dt v = a
                  * 2 static pillars   -> com_obstacle_hocbfs         (relative degree 2)
                  * 3 pedestrians      -> com_moving_obstacle_hocbfs  (time-varying: dh/dt in the QP)
    locomotion: Unitree's pretrained walking policy tracks the integrated velocity command,
                turning to face it (heading follower)
    plant:      MJX G1 (12-DoF legs, PD at 500 Hz)

Pedestrians are *kinematic* and certificate-only: they walk with known constant velocity
``p_i(t) = p_i(0) + v_i t``, are not MuJoCo bodies (the G1 cannot physically bump into them),
and are drawn into the GIF/viewer as moving capsules. Their barrier is
``h_i(t, x) = ||(com - p_i(t)) / r||^2 - 1`` with ``r = PED_RADIUS + ROBOT_RADIUS``; the
rectifier lifts it to relative degree 2 and carries ``dh/dt`` -- a pedestrian walking at a
standing robot already makes the QP back it away. Encounters: one head-on (leg 2), one crossing
from the left (leg 1) and one slow crossing from the right (leg 3); both pillars sit within
the keep-out of the straight nominal legs, so every obstacle forces the CBF to act.

Safety measure: ``h_min`` over all obstacles (>= 0 means the CoM never entered a keep-out
disc). The robust bound is the measured tracking error ``||v_com - v_safe||`` of the gait *in
this scenario* (``--robust``; 0 = vanilla): DI 0.28 sits above the max observed under it (the
self-consistent claim), SI 0.18 is the p95 under it (the SI max is a spike at the waypoint
switches, where its velocity command jumps). Measured (CPU, 45 s horizon, seed 0):

    model  CBF      bound  h_min   closest ped [m]  W1/W2/W3 reached [s]  tracking mean/p95/max [m/s]
    di     vanilla  --     -0.048  0.73             13.1 / 24.4 / 32.8    0.078 / 0.147 / 0.271
    di     robust   0.15   +0.811  1.01             13.9 / 27.0 / 35.1    0.076 / 0.138 / 0.253
    di     robust   0.28*  +1.979  1.30             14.7 / 30.2 / 37.8    0.075 / 0.136 / 0.251  (* default)
    si     vanilla  --     -0.090  0.70             12.4 / 28.0 / 35.0    0.100 / 0.241 / 0.643
    si     robust   0.18*  +0.616  0.95             13.4 / 36.3 / 43.3    0.088 / 0.169 / 0.647  (* default; so
                                                                          slow that pedestrian 2 has passed by leg 3)

The vanilla rows show the point: the gait tracks the certified command only approximately, so
the vanilla CBF lets the CoM graze the keep-out (h_min < 0 at pillar 2); the robust CBF holds
every barrier with margin -- at the price of a wider berth and a slower crossing. Note the
robust QP is per-certificate: the margin ``||dh_i/dx|| * bound`` is subtracted from constraint
``i`` only (fixed alongside this example -- with the previous stacked-norm bug five barriers
made the robust QP infeasible at bounds the single-obstacle ``g1_navigate`` handled).

    python examples/mujoco/g1_plaza.py [--reduced-model di|si] [--robust B] [--duration T] [--gif] [--view]
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
from cbfkit.planners import waypoint_route
from cbfkit.systems.mujoco.reduced_order import (
    com_moving_obstacle_barriers,
    com_moving_obstacle_hocbfs,
    com_obstacle_barriers,
    com_obstacle_hocbfs,
    embedded_double_integrator,
    embedded_single_integrator,
    moving_obstacle_position,
    safe_locomotion_controller,
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
PEDESTRIANS = jnp.array(  # (x0, y0, vx, vy): known constant velocity, kinematic
    [
        [6.5, -1.4, -0.32, 0.24],  # head-on along leg 2, meets the robot near x ~ 4
        [8.5, -3.9, -0.025, 0.14],  # strolls across leg 3 from the right, near x ~ 7.8 at t ~ 28 s
        [0.5, 3.0, 0.25, -0.40],  # cuts across leg 1 from the front-left near t ~ 6 s
    ]
)
PED_RADIUS = 0.30
ROBOT_RADIUS = 0.35  # planar inflation so "CoM outside the disc" => "body clear"
V_MAX = 0.5
A_MAX = 1.0  # double-integrator model: acceleration limit [m/s^2]
_TAG = ["run"]  # results filename tag, set per run

R_PILLAR = PILLAR_RADIUS + ROBOT_RADIUS
R_PED = PED_RADIUS + ROBOT_RADIUS
OBSTACLE_NAMES = [f"pillar {i+1}" for i in range(len(PILLARS))] + [
    f"pedestrian {i+1}" for i in range(len(PEDESTRIANS))
]


def build(seed: int = 0, robust_bound: float = 0.0, reduced_model: str = "di"):
    """Plant, policy, planner, nominal and safety controller for the plaza scenario."""
    plant = make_g1_12dof_plant()  # 12-DoF legs, PD at 500 Hz, dt 0.02
    loco = UnitreeG1WalkPolicy().as_controller()
    x0 = x0_standing(plant)
    pelvis_body = int(plant.mj_model.body("pelvis").id)
    ci = plant.com_indices

    ped_p0 = [PEDESTRIANS[i, :2] for i in range(len(PEDESTRIANS))]
    ped_v = [PEDESTRIANS[i, 2:] for i in range(len(PEDESTRIANS))]
    pillar_ell = [(R_PILLAR, R_PILLAR)] * len(PILLARS)
    ped_ell = [(R_PED, R_PED)] * len(PEDESTRIANS)
    if reduced_model == "di":
        dyn = embedded_double_integrator(plant.state_dim, ci)
        barriers = concatenate_certificates(
            com_obstacle_hocbfs(plant, PILLARS, pillar_ell, class_k_gain=1.0),
            com_moving_obstacle_hocbfs(plant, ped_p0, ped_v, ped_ell, class_k_gain=1.0),
        )
        limits = jnp.array([A_MAX, A_MAX])
    else:
        dyn = embedded_single_integrator(plant.state_dim, ci)
        barriers = concatenate_certificates(
            com_obstacle_barriers(plant, PILLARS, pillar_ell, class_k_gain=1.0),
            com_moving_obstacle_barriers(plant, ped_p0, ped_v, ped_ell, class_k_gain=1.0),
        )
        limits = jnp.array([V_MAX, V_MAX])
    if robust_bound > 0.0:
        cbf_qp = robust_cbf_clf_qp_controller(
            control_limits=limits,
            dynamics_func=dyn,
            barriers=barriers,
            disturbance_norm=2,
            disturbance_norm_bound=float(robust_bound),
        )
    else:
        cbf_qp = vanilla_cbf_clf_qp_controller(
            control_limits=limits, dynamics_func=dyn, barriers=barriers
        )
    if reduced_model == "di":
        safe = safe_locomotion_controller_di(cbf_qp, loco, plant, plant.dt, v_max=V_MAX)
    else:
        safe = safe_locomotion_controller(cbf_qp, loco)

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


def barrier_values(com: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``h`` per obstacle over the run, shape ``(len(t), n_pillars + n_pedestrians)``."""
    hs = []
    for i in range(len(PILLARS)):
        d = com - np.asarray(PILLARS[i])
        hs.append((d**2).sum(1) / R_PILLAR**2 - 1)
    for i in range(len(PEDESTRIANS)):
        p = np.asarray(PEDESTRIANS[i, :2])[None] + np.asarray(PEDESTRIANS[i, 2:])[None] * t[:, None]
        d = com - p
        hs.append((d**2).sum(1) / R_PED**2 - 1)
    return np.stack(hs, axis=1)


def waypoint_arrivals(com: np.ndarray) -> list:
    """First step index at which each waypoint (in order) is reached, or None."""
    out, start = [], 0
    for wp in np.asarray(WAYPOINTS):
        d = np.linalg.norm(com[start:] - wp, axis=1)
        hit = np.flatnonzero(d < WAYPOINT_RADIUS)
        if hit.size == 0:
            out.append(None)
            break
        start += int(hit[0])
        out.append(start)
    while len(out) < len(WAYPOINTS):
        out.append(None)
    return out


DEFAULT_ROBUST_BOUND = {"di": 0.28, "si": 0.18}  # measured in this scenario, see the docstring
DEFAULT_DURATION = 45.0


def main(
    duration=DEFAULT_DURATION,
    seed=0,
    gif=False,
    view=False,
    robust_bound=None,
    reduced_model="di",
):
    if robust_bound is None:  # measured per model in this scenario, see the module docstring
        robust_bound = DEFAULT_ROBUST_BOUND[reduced_model]
    plant, x0, pelvis_body, planner, nominal, controller = build(seed, robust_bound, reduced_model)
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
    H = barrier_values(com, t)
    cd = res.controller_data
    v_nom = np.asarray(cd.get("sub_data_v_nom")) if "sub_data_v_nom" in cd else None
    v_safe = np.asarray(cd.get("sub_data_v_safe")) if "sub_data_v_safe" in cd else None
    hh = states[:, 2]  # pelvis height
    q = states[:, 3:7]  # pelvis quaternion (w, x, y, z): body z-axis . world z
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    arrivals = waypoint_arrivals(com)

    print(f"{steps} steps in {wall:.1f}s")
    print(f"h(x) min over run: {H.min():.3f}   (>= 0 means the CoM never entered a keep-out disc)")
    for name, hcol in zip(OBSTACLE_NAMES, H.T):
        k = int(np.argmin(hcol))
        print(f"  {name:13s} h_min {hcol[k]:+.3f} at t={t[k]:.1f}s")
    ped_d = np.sqrt((H[:, len(PILLARS) :] + 1) * R_PED**2)
    print(f"closest CoM-pedestrian distance: {ped_d.min():.2f} m (keep-out {R_PED:.2f})")
    print(
        "waypoints reached at: "
        + ", ".join("--" if k is None else f"{k*plant.dt:.1f}s" for k in arrivals)
        + ("" if arrivals[-1] is not None else "  (route NOT completed)")
    )
    print(f"pelvis height min {hh.min():.2f}, upright min {up.min():.2f}")
    if v_nom is not None and v_safe is not None:
        print(
            f"CBF active (|v_safe - v_nom| > 1e-3) on {np.mean(np.linalg.norm(v_safe - v_nom, axis=1) > 1e-3)*100:.0f}% of steps"
        )
        n_live = arrivals[-1] if arrivals[-1] is not None else len(com)
        v_com = np.gradient(com[:n_live], plant.dt, axis=0)
        err = np.linalg.norm(v_com - v_safe[:n_live], axis=1)
        print(
            f"tracking error ||v_com - v_safe||: mean {err.mean():.3f}, p95 {np.percentile(err, 95):.3f}, "
            f"max {err.max():.3f} m/s  (use as --robust bound)"
        )
    if TEST_MODE:
        return float(H.min())
    _TAG[0] = f"{reduced_model}_{'robust' if robust_bound > 0 else 'vanilla'}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # The simulation latches after completion (state frozen); plot/render only the live part.
    n = len(com) if arrivals[-1] is None else min(len(com), arrivals[-1] + int(1.0 / plant.dt))
    _plot(
        com[:n],
        t[:n],
        H[:n],
        None if v_nom is None else v_nom[:n],
        None if v_safe is None else v_safe[:n],
        hh[:n],
    )
    if gif:
        path = os.path.join(RESULTS_DIR, f"g1_plaza_{_TAG[0]}.gif")
        render_gif(plant, states[:n], path, track_body=pelvis_body, markers=_markers, distance=5.5)
    if view:
        replay_in_viewer(plant, states[:n], markers=_markers)
    return float(H.min())


def _plot(com, t, H, v_nom, v_safe, hh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [2, 1.2, 1]})
    ax = axes[0]
    for i, p in enumerate(np.asarray(PILLARS)):
        ax.add_patch(plt.Circle(p, PILLAR_RADIUS, color="crimson", alpha=0.7))
        ax.add_patch(plt.Circle(p, R_PILLAR, color="crimson", fill=False, ls="--", lw=0.8))
    for i, row in enumerate(np.asarray(PEDESTRIANS)):
        p0, v = row[:2], row[2:]
        traj = p0[None] + v[None] * t[:, None]
        ax.plot(traj[:, 0], traj[:, 1], ":", color="tab:blue", lw=1)
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
    ax.set_xlim(-1, 10)
    ax.set_ylim(-4.2, 3.5)
    ax.legend(loc="lower right")
    ax.set_title("plaza: CoM path, pillars (red), pedestrians (blue), waypoints (green)")
    fig.colorbar(sc, ax=ax, label="t [s]", shrink=0.7)
    for name, hcol in zip(OBSTACLE_NAMES, H.T):
        axes[1].plot(t, hcol, lw=1.5, label=name)
    axes[1].axhline(0, color="k", ls=":")
    axes[1].set_ylim(-0.5, 8)
    axes[1].set_title("barriers h_i(t, x) (>= 0 safe)")
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


def _markers(scn, k, t):
    """Pillars, pedestrians at time t, keep-out rings, waypoints and goal (visual-only geoms)."""
    import mujoco

    cyl, sph, cap = (
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
    )
    for p in np.asarray(PILLARS):
        add_marker(scn, cyl, [PILLAR_RADIUS, 0.6, 0], [*p, 0.6], [0.85, 0.15, 0.2, 1.0])
        add_marker(scn, cyl, [R_PILLAR, 0.005, 0], [*p, 0.005], [0.85, 0.15, 0.2, 0.25])
    for row in np.asarray(PEDESTRIANS):
        p = np.asarray(moving_obstacle_position(row[:2], row[2:], t))
        add_marker(scn, cap, [PED_RADIUS * 0.6, 0.55, 0], [*p, 0.85], [0.2, 0.4, 0.9, 1.0])
        add_marker(scn, sph, [PED_RADIUS * 0.5, 0, 0], [*p, 1.55], [0.2, 0.4, 0.9, 1.0])
        add_marker(scn, cyl, [R_PED, 0.005, 0], [*p, 0.005], [0.2, 0.4, 0.9, 0.25])
    for i, wp in enumerate(np.asarray(WAYPOINTS)):
        last = i == len(WAYPOINTS) - 1
        r = GOAL_RADIUS if last else 0.12
        add_marker(scn, sph, [r, 0, 0], [*wp, r], [0.1, 0.8, 0.2, 0.6 if last else 0.9])


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    p.add_argument(
        "--robust",
        type=float,
        default=None,
        help="robust CBF with this ||v_com - v_safe|| bound in m/s (default: measured per model); 0 = vanilla",
    )
    p.add_argument(
        "--reduced-model",
        default="di",
        choices=["si", "di"],
        help="CBF reduced-order model: single integrator (velocity command) or double integrator (acceleration, HOCBF)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    if a.view:
        relaunch_under_mjpython_if_needed()
    main(a.duration, a.seed, a.gif, a.view, a.robust, a.reduced_model)
