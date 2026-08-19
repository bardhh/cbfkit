"""Unitree G1 crosses a Shibuya-style scramble: ~30 pedestrians released on green, flowing in
six streams, while the robot takes the diagonal -- tracked-agent HOCBFs at crowd scale.

Same stack as ``g1_plaza.py`` (command-side double integrator, distance-shaped keep-out
barriers on *tracked agents* with constant-velocity prediction, robust CBF-QP on the in-repo
PDIPM, Unitree's walking policy, MJX G1), scaled up: ``N_PED`` social-force pedestrians
(``SocialForceCrowd``) start 0-30 m behind the kerbs of a 12 x 12 m intersection (so they
arrive throughout the robot's crossing) and cross
W<->E, S<->N and along both diagonals at 0.8-1.3 m/s -- faster than the robot (0.5 m/s) --
reacting to the robot and to each other. The robot crosses SW -> NE (17 m).

What to look at: ``h_min`` over all pedestrians (>= 0: no keep-out disc ever entered),
near-misses, the crossing time and the fraction of time the robot stood waiting for a gap
(the "freezing robot" regime), plus solver health -- with ~30 relative-degree-2 barriers and
robust margins the hard-constrained QP can become infeasible in a crush; ``--relax`` turns
the barrier constraints soft (``relaxable_cbf``) and the slack used is then reported as the
certificate's model-level violation. Measured (CPU, seed 0, 40 pedestrians, 60 s horizon
unless noted; "closest" = CoM-pedestrian, keep-out 0.65 m; "slack" = fraction of steps a barrier
row needed slack):

    constraints  CBF      bound  result
    hard         vanilla  --     QP INFEASIBLE at t = 8.5 s (sim stops): two pedestrians closing at ~1 m/s
    hard         robust   0.15   QP INFEASIBLE at t = 13.4 s (sim stops)  from two sides, |a| <= 1 m/s^2 can't
                                 satisfy both under constant-velocity prediction -- a true crush, not a solver issue
    soft*        vanilla  --     crossed 49.2 s; h_min +0.083 (closest 0.70 m); 16 pedestrians within 1.5 m,
                                 1 near-miss; slack on 10 % of steps (max 0.29); waiting 0 %; tracking 0.088/0.190/0.342
    soft         robust   0.15   crossed 65.0 s (75 s horizon); h_min -0.051 (closest 0.62 m, 38 steps inside the
                                 keep-out, no contact); slack on 18.5 % (max 0.95): the margins are simply eaten by slack

    (* default)

Read it this way: in a crowd that is faster than the robot, the barrier constraints cannot be
hard -- the QP is infeasible within seconds whatever the bound -- so nothing here is a
*certificate*; the soft-constrained QP is a safety *filter* whose outcome (h_min, near-misses,
slack) is what gets reported, and the robust margin buys nothing once slack is active. The
robot still crosses a 40-pedestrian scramble without contact while the pedestrians (social
force) yield around it; the one squeeze (t ~ 36 s, h 0.08) is visible in the plot.

    python examples/mujoco/g1_scramble.py [--robust B] [--pedestrians N] [--relax] [--duration T] [--seed S] [--gif] [--view]
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
from cbfkit.controllers.cbf_clf import robust_cbf_clf_qp_controller, vanilla_cbf_clf_qp_controller
from cbfkit.optimization.quadratic_program.solver_registry import get_solver
from cbfkit.systems.mujoco.crowd import SocialForceCrowd
from cbfkit.systems.mujoco.reduced_order import (
    com_agent_hocbfs,
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
HALF = 6.0  # the intersection is [-HALF, HALF]^2; kerbs at +-HALF
START = jnp.array([-HALF, -HALF])
GOAL = jnp.array([HALF, HALF])
GOAL_RADIUS = 0.4
N_PED = 8 if TEST_MODE else 40
PED_SPEED_RANGE = (0.8, 1.3)  # m/s -- real pedestrians, faster than the robot
PED_RADIUS = 0.30
PED_REPULSION_RANGE = 0.4
RELEASE_DEPTH = 30.0  # pedestrians start 0-30 m behind the kerbs -> arrivals spread over ~30 s
ARRIVE_RADIUS = 1.0  # they stop at their goals (6 m past the far kerb)
ROBOT_RADIUS = 0.35
R_PED = PED_RADIUS + ROBOT_RADIUS
V_MAX = 0.5
A_MAX = 1.0
BARRIER_SHAPE = "distance"
DEFAULT_ROBUST_BOUND = 0.0  # see the docstring table: in a crush robust margins are eaten by slack
DEFAULT_RELAX = True  # hard barrier constraints are infeasible within ~10 s in this crowd
DEFAULT_DURATION = 60.0
_TAG = ["run"]

# streams: (start side, goal side) in intersection coordinates; weights favour the straight ones
_STREAMS = [
    ("W", "E", 5),
    ("E", "W", 5),
    ("S", "N", 5),
    ("N", "S", 5),
    ("SW", "NE", 2),
    ("NE", "SW", 2),
    ("NW", "SE", 2),
    ("SE", "NW", 2),
]
_ANCHOR = {
    "W": (-1, 0),
    "E": (1, 0),
    "S": (0, -1),
    "N": (0, 1),
    "SW": (-1, -1),
    "NE": (1, 1),
    "NW": (-1, 1),
    "SE": (1, -1),
}


def make_crowd(n_ped: int, seed: int):
    """Starts/goals/speeds for ``n_ped`` pedestrians drawn from the streams (seeded)."""
    rng = np.random.default_rng(seed + 1000)
    sides = [s for s in _STREAMS for _ in range(s[2])]
    starts, goals, speeds = [], [], []
    for i in range(n_ped):
        a, b, _ = sides[i % len(sides)]
        ua = np.asarray(_ANCHOR[a], float)
        ub = np.asarray(_ANCHOR[b], float)
        ua_n = ua / np.linalg.norm(ua)
        ub_n = ub / np.linalg.norm(ub)
        perp = np.array([-ua_n[1], ua_n[0]])
        back = rng.uniform(0.0, RELEASE_DEPTH)  # metres behind the kerb at t = 0: staggers arrivals
        lateral = rng.uniform(-2.5, 2.5) if len(a) == 1 else rng.uniform(-1.0, 1.0)
        start = ua_n * (HALF * np.linalg.norm(ua) + back) + perp * lateral
        goal = ub_n * (HALF * np.linalg.norm(ub) + 6.0) + perp * (lateral + rng.uniform(-1.0, 1.0))
        starts.append(start)
        goals.append(goal)
        speeds.append(rng.uniform(*PED_SPEED_RANGE))
    return np.asarray(starts), np.asarray(goals), np.asarray(speeds)


def build(seed: int = 0, robust_bound: float = 0.0, n_ped: int = N_PED, relax: bool = False):
    plant = make_g1_12dof_plant()
    loco = UnitreeG1WalkPolicy().as_controller()
    x0 = x0_standing(plant)
    # Place the robot at START (the XML puts it at the origin): shift the pelvis x, y.
    x0 = x0.at[0:2].add(START).at[plant.com_indices[0] : plant.com_indices[0] + 2].add(START)
    pelvis_body = int(plant.mj_model.body("pelvis").id)
    ci = plant.com_indices
    starts, goals, speeds = make_crowd(n_ped, seed)
    crowd = SocialForceCrowd(
        starts,
        goals,
        speeds,
        ped_radius=PED_RADIUS,
        agent_radius=ROBOT_RADIUS,
        repulsion_range=PED_REPULSION_RANGE,
        arrive_radius=ARRIVE_RADIUS,
    )
    dyn = embedded_double_integrator(plant.state_dim, ci, n_agents=n_ped)
    barriers = com_agent_hocbfs(
        plant, n_ped, [(R_PED, R_PED)] * n_ped, class_k_gain=1.0, shape=BARRIER_SHAPE
    )
    kw = dict(
        control_limits=jnp.array([A_MAX, A_MAX]),
        dynamics_func=dyn,
        barriers=barriers,
        solver=get_solver("fast"),
    )
    if relax:
        kw.update(relaxable_cbf=True, slack_penalty_cbf=1e3, slack_bound_cbf=10.0)
    if robust_bound > 0.0:
        cbf_qp = robust_cbf_clf_qp_controller(
            disturbance_norm=2, disturbance_norm_bound=float(robust_bound), **kw
        )
    else:
        cbf_qp = vanilla_cbf_clf_qp_controller(**kw)
    safe = safe_locomotion_controller_di(cbf_qp, loco, plant, plant.dt, v_max=V_MAX, agents=crowd)

    def controller(t, x, v_nom, key, data):
        u, d = safe(t, x, v_nom, key, data)
        reached = jnp.linalg.norm(x[ci[0] : ci[0] + 2] - GOAL) < GOAL_RADIUS
        return u, d._replace(complete=d.complete | reached)

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]

    def nominal(t, x, key, ref):
        com = x[ci[0] : ci[0] + 2]
        v = 1.0 * (jnp.asarray(ref)[:2] - com)
        speed = jnp.linalg.norm(v)
        v = jnp.where(speed > V_MAX, v * (V_MAX / (speed + 1e-9)), v)
        return v, ControllerData()

    return plant, x0, pelvis_body, nominal, controller, crowd


def main(
    duration=DEFAULT_DURATION,
    seed=0,
    gif=False,
    view=False,
    robust_bound=None,
    n_ped=N_PED,
    relax=None,
):
    if robust_bound is None:
        robust_bound = DEFAULT_ROBUST_BOUND
    if relax is None:
        relax = DEFAULT_RELAX
    plant, x0, pelvis_body, nominal, controller, crowd = build(seed, robust_bound, n_ped, relax)
    steps = 5 if TEST_MODE else int(round(duration / plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        plant=plant,
        planner_data=PlannerData.from_constant(GOAL),
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    wall = time.time() - t0
    S = np.asarray(res["states"])
    ci = plant.com_indices
    com = S[:, ci[0] : ci[0] + 2]
    t = np.arange(len(com)) * plant.dt
    cd = res.controller_data
    agents = np.asarray(cd["sub_data_agents"])  # (T, N, 4)
    d = np.linalg.norm(com[:, None, :] - agents[:, :, :2], axis=2)  # (T, N)
    H = d / R_PED - 1.0
    status = np.asarray(cd["sub_data_solver_status"])
    v_nom = np.asarray(cd["sub_data_v_nom"])
    v_safe = np.asarray(cd["sub_data_v_safe"])
    dist_goal = np.linalg.norm(com - np.asarray(GOAL), axis=1)
    hit = np.flatnonzero(dist_goal < GOAL_RADIUS)
    n_live = int(hit[0]) if hit.size else len(com)
    err_steps = np.flatnonzero(np.asarray(cd["error"]))
    stopped_at = int(err_steps[0]) if err_steps.size else None
    hh, q = S[:, 2], S[:, 3:7]
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    speed = np.linalg.norm(np.gradient(com[:n_live], plant.dt, axis=0), axis=1)
    waiting = float(np.mean(speed < 0.1)) if n_live > 1 else 0.0
    near = int(np.sum(d[:n_live].min(0) < R_PED + 0.15))
    slack = None
    if relax and "sol" in cd:
        sol = np.asarray(cd["sol"])
        slack = sol[:n_live, 2:]  # slack columns follow the 2 controls
    ped_speed = np.linalg.norm(agents[:n_live, :, 2:], axis=2)
    pp = np.linalg.norm(agents[:n_live, :, None, :2] - agents[:n_live, None, :, :2], axis=3)
    pp[:, np.arange(n_ped), np.arange(n_ped)] = np.inf

    print(f"{steps} steps in {wall:.1f}s  ({n_ped} pedestrians)")
    print(
        f"h(x) min over run: {H[:n_live].min():.3f}   (>= 0 means no pedestrian keep-out disc was entered)"
    )
    print(
        f"closest CoM-pedestrian distance: {d[:n_live].min():.2f} m (keep-out {R_PED:.2f}); "
        f"near-misses (< keep-out + 0.15 m): {near} of {n_ped}; pedestrians that came within 1.5 m: {int(np.sum(d[:n_live].min(0) < 1.5))}"
    )
    print(
        f"crossed at t={n_live*plant.dt:.1f}s"
        if hit.size
        else "crossing NOT completed" f" (distance to goal at the end {dist_goal[-1]:.2f} m)"
    )
    print(
        f"robot waiting (|v_com| < 0.1 m/s) {waiting*100:.0f}% of the live steps; "
        f"CBF active on {np.mean(np.linalg.norm(v_safe[:n_live] - v_nom[:n_live], axis=1) > 1e-3)*100:.0f}%"
    )
    print(f"pelvis height min {hh[:n_live].min():.2f}, upright min {up[:n_live].min():.2f}")
    print(
        f"QP: {int(np.sum(status[:n_live] != 1))} non-converged steps of {n_live}"
        + (
            ""
            if stopped_at is None
            else f"; SIMULATION STOPPED on controller error at t={stopped_at*plant.dt:.1f}s"
        )
    )
    if slack is not None:
        print(
            f"relaxed barriers: slack > 1e-3 on {np.mean(slack.max(1) > 1e-3)*100:.1f}% of steps, max slack {slack.max():.3f}"
        )
    print(
        f"crowd: mean speed ratio {float((ped_speed / np.asarray(crowd.speeds)[None]).mean()):.2f}, "
        f"min pedestrian-pedestrian distance {pp.min():.2f} m"
    )
    n_vio = int(np.sum((H[:n_live] < 0).any(1)))
    if n_vio:
        print(f"  keep-out violated on {n_vio} steps (worst h {H[:n_live].min():.3f})")
    if v_safe is not None:
        v_com = np.gradient(com[:n_live], plant.dt, axis=0)
        err = np.linalg.norm(v_com - v_safe[:n_live], axis=1)
        print(
            f"tracking error ||v_com - v_safe||: mean {err.mean():.3f}, p95 {np.percentile(err, 95):.3f}, max {err.max():.3f} m/s"
        )
    if TEST_MODE:
        return float(H[:n_live].min())
    _TAG[0] = f"{'relaxed_' if relax else ''}{'robust' if robust_bound > 0 else 'vanilla'}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n = min(len(com), n_live + int(1.0 / plant.dt))
    _plot(com[:n], t[:n], H[:n], agents[:n], v_nom[:n], v_safe[:n], crowd)
    markers = _make_markers(agents)
    if gif:
        path = os.path.join(RESULTS_DIR, f"g1_scramble_{_TAG[0]}.gif")
        render_gif(
            plant,
            S[:n],
            path,
            track_body=pelvis_body,
            markers=markers,
            distance=7.0,
            elevation=-35.0,
        )
    if view:
        replay_in_viewer(plant, S[:n], markers=markers)
    return float(H[:n_live].min())


def _plot(com, t, H, agents, v_nom, v_safe, crowd):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4)
    snaps = [int(f * (len(t) - 1)) for f in (0.15, 0.4, 0.65, 0.9)]
    for j, k in enumerate(snaps):
        ax = fig.add_subplot(gs[0, j])
        ax.add_patch(
            plt.Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False, ls="--", color="gray")
        )
        for i in range(agents.shape[1]):
            ax.add_patch(plt.Circle(agents[k, i, :2], PED_RADIUS, color="tab:blue", alpha=0.6))
            ax.arrow(
                *agents[k, i, :2],
                *(agents[k, i, 2:] * 0.6),
                head_width=0.15,
                color="tab:blue",
                alpha=0.6,
            )
        ax.plot(com[: k + 1, 0], com[: k + 1, 1], "-", color="tab:green", lw=2)
        ax.add_patch(plt.Circle(com[k], ROBOT_RADIUS, color="tab:green"))
        ax.add_patch(plt.Circle(com[k], R_PED, color="tab:green", fill=False, ls=":"))
        ax.plot(*np.asarray(GOAL), "g*", ms=12)
        ax.set_aspect("equal")
        ax.set_xlim(-HALF - 3, HALF + 3)
        ax.set_ylim(-HALF - 3, HALF + 3)
        ax.set_title(f"t = {t[k]:.1f} s")
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t, H.min(1), lw=1.5, label="min_i h_i (closest pedestrian)")
    ax.axhline(0, color="k", ls=":")
    ax.set_ylim(-0.3, 3)
    ax.set_xlabel("t [s]")
    ax.set_title("closest-pedestrian barrier (>= 0 safe)")
    ax.legend()
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(t, np.linalg.norm(v_nom, axis=1), "--", label="|v_nom|")
    ax.plot(t, np.linalg.norm(v_safe, axis=1), label="|v_safe|")
    v_com = np.gradient(com, t[1] - t[0], axis=0)
    ax.plot(t, np.linalg.norm(v_com, axis=1), lw=0.6, alpha=0.6, label="|v_com| (measured)")
    ax.set_xlabel("t [s]")
    ax.set_title("CoM speed: nominal vs certified vs measured")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"g1_scramble_{_TAG[0]}.png")
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def _make_markers(agents):
    import mujoco

    sph, cap, cyl = (
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    )
    last = len(agents) - 1
    corners = [np.array(c, float) * HALF for c in ((1, 1), (1, -1), (-1, 1), (-1, -1))]

    def markers(scn, k, t):
        for p in agents[min(k, last), :, :2]:
            add_marker(scn, cap, [PED_RADIUS * 0.6, 0.55, 0], [*p, 0.85], [0.2, 0.4, 0.9, 1.0])
            add_marker(scn, sph, [PED_RADIUS * 0.5, 0, 0], [*p, 1.55], [0.2, 0.4, 0.9, 1.0])
        for c in corners:  # kerb markers
            add_marker(scn, cyl, [0.12, 0.3, 0], [*c, 0.3], [0.6, 0.6, 0.6, 1.0])
        add_marker(
            scn, sph, [GOAL_RADIUS, 0, 0], [*np.asarray(GOAL), GOAL_RADIUS], [0.1, 0.8, 0.2, 0.6]
        )

    return markers


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    p.add_argument("--robust", type=float, default=None, help="robust bound (m/s); 0 = vanilla")
    p.add_argument("--pedestrians", type=int, default=N_PED)
    p.add_argument(
        "--relax",
        dest="relax",
        action="store_true",
        default=None,
        help="soft barrier constraints (slack, penalised) -- the default here",
    )
    p.add_argument("--hard", dest="relax", action="store_false", help="hard barrier constraints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    if a.view:
        relaunch_under_mjpython_if_needed()
    main(a.duration, a.seed, a.gif, a.view, a.robust, a.pedestrians, a.relax)
