"""Unitree G1 walks to a goal past an obstacle: reduced-order CBF on the CoM + a walking policy.

Pipeline (all inside one CBFKit `execute(plant=...)` call):

    goal ->  nominal: v_nom = k (p_goal - com)          (2-D, saturated)
         ->  safety: vanilla CBF-QP on  d/dt com = v    (ellipsoidal keep-out around the obstacle,
                                                         evaluated at plant.com_indices)  -> v_safe
         ->  locomotion: tracks v_safe                    -> joint-position targets
         ->  plant: MJX G1

Locomotion layers (``--locomotion``):
  policy  (default) Unitree's pretrained ``unitree_rl_gym`` G1 walking policy (12-DoF legs, LSTM),
                    read without torch and evaluated in JAX; PD torques at 500 Hz. Walks and steers.
  mpc               the in-repo MJX sampling MPC on the 29-DoF model (in-house gait search, see g1_walk_trial.py): forward
                    locomotion with stumbles; kept for comparison.

Reduced-order models (``--reduced-model``):
  di  (default) command-side double integrator on the CoM: the CBF-QP filters an *acceleration*
                (high-order barrier via ``rectify_relative_degree``); the velocity command is its
                integral -- smooth, |a| <= 1 m/s^2. Sounder model of a walking CoM; its smooth
                commands cut the gait's worst-case tracking error 3x (max 0.17 vs 0.49 m/s).
  si            single integrator: the CBF-QP filters the velocity command directly (the stock
                ellipsoidal barrier, unmodified, pointed at plant.com_indices).

The CBF certifies the *commanded* CoM velocity; the gait tracks it approximately. The default is
therefore the **robust** CBF-QP with a disturbance bound taken from the measured tracking error
||v_com - v_safe|| of the policy gait *in this scenario* (``--robust`` picks it: di 0.18 = above the
observed max, si 0.16 = p95 since its max 0.49 is a startup spike; 0 = vanilla). The robot turns
to face its commanded velocity (heading follower in the policy adapter), so it walks around the
obstacle rather than sidestepping. Measured (CPU, 20 s, goal radius 0.3 m):

    model  CBF      bound   h_min   closest [m]  goal [s]   tracking err mean/p95/max [m/s]
    si     vanilla  --      -0.037  0.69         14.1       0.088 / 0.156 / 0.488
    si     robust   0.16    +0.481  0.85         16.1       0.084 / 0.148 / 0.488
    di     vanilla  --      -0.051  0.68         15.5       0.078 / 0.140 / 0.174
    di     robust   0.14    +0.724  0.92         18.3       0.074 / 0.131 / 0.162
    di     robust   0.18*   +1.007  0.99         19.3       0.073 / 0.129 / 0.165   (* default)

    python examples/mujoco/g1_navigate.py [--reduced-model di|si] [--robust B] [--locomotion policy|mpc] [--gif] [--view]
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
from cbfkit.controllers.mjx_sampling_mpc import SamplingMpc
from cbfkit.systems.mujoco import MujocoPlant
from cbfkit.systems.mujoco.g1 import G1, load_g1, walk_costs
from cbfkit.systems.mujoco.reduced_order import (
    com_obstacle_barriers,
    com_obstacle_hocbfs,
    embedded_double_integrator,
    embedded_single_integrator,
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
from cbfkit.utils.user_types import ControllerData

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/cbfkit/jax"))

GOAL = jnp.array([4.0, 0.0])
OBSTACLE = jnp.array([2.0, 0.0])  # centre, on the straight line to the goal
OBSTACLE_RADIUS = 0.35
ROBOT_RADIUS = 0.35  # planar inflation so "CoM outside the ellipse" => "body clear"
V_MAX = 0.5
A_MAX = 1.0  # double-integrator model: acceleration limit [m/s^2]
GOAL_RADIUS = 0.3
_TAG = ["run"]  # results filename tag, set per run


def build(
    locomotion: str = "policy",
    num_samples: int = 256,
    iterations: int = 2,
    seed: int = 0,
    robust_bound: float = 0.0,
    reduced_model: str = "si",
):
    if locomotion == "policy":
        sim_plant = make_g1_12dof_plant()  # 12-DoF legs, PD at 500 Hz, dt 0.02
        loco = UnitreeG1WalkPolicy().as_controller()
        x0 = x0_standing(sim_plant)
        pelvis_body = int(sim_plant.mj_model.body("pelvis").id)
        return (
            sim_plant,
            loco,
            x0,
            pelvis_body,
            _finish(sim_plant, loco, robust_bound, reduced_model),
        )

    sim_plant = MujocoPlant(load_g1(sim=True), substeps=2)
    mpc_plant = MujocoPlant(load_g1())
    g1 = G1(sim_plant.mj_model)

    # Locomotion: best-so-far walk configuration (from the in-house gait search).
    running, terminal = walk_costs(
        g1,
        w_velocity=10.0,
        w_height=10.0,
        w_balance=5.0,
        w_yaw_rate=5.0,
        w_fall=100.0,
        w_qvel=0.01,
        w_gait=50.0,
        gait_freq=1.5,
        gait_swing_height=0.08,
    )
    noise = np.full(mpc_plant.nu, 0.3)
    noise[12:15] *= 0.5  # waist
    noise[15:] *= 0.3  # arms
    mpc = SamplingMpc(
        mpc_plant,
        running,
        terminal,
        num_samples=num_samples,
        plan_horizon=0.32,
        noise_level=noise,
        temperature=0.1,
        num_knots=4,
        spline_type="zero",
        iterations=iterations,
        seed=seed,
    )

    x0 = g1.x_stand(sim_plant)
    return (
        sim_plant,
        mpc.as_controller(),
        x0,
        g1.pelvis_body,
        _finish(sim_plant, mpc.as_controller(), robust_bound, reduced_model),
    )


def _finish(sim_plant, loco, robust_bound: float = 0.0, reduced_model: str = "si"):
    """Safety layer + nominal, identical for every locomotion layer.

    ``reduced_model``: ``"si"`` -- single integrator on the CoM, the CBF-QP filters
    the velocity command directly; ``"di"`` -- command-side double integrator, the
    CBF-QP (high-order barrier via ``rectify_relative_degree``) filters an
    acceleration and the velocity command is its integral (smooth, bounded by
    ``A_MAX``). ``robust_bound > 0`` uses the robust CBF-QP with that 2-norm
    disturbance bound -- the measured tracking error of the gait.
    """
    r = OBSTACLE_RADIUS + ROBOT_RADIUS
    if reduced_model == "di":
        dyn = embedded_double_integrator(sim_plant.state_dim, sim_plant.com_indices)
        barriers = com_obstacle_hocbfs(sim_plant, [OBSTACLE], [(r, r)], class_k_gain=1.0)
        limits = jnp.array([A_MAX, A_MAX])
    else:
        dyn = embedded_single_integrator(sim_plant.state_dim, sim_plant.com_indices)
        barriers = com_obstacle_barriers(sim_plant, [OBSTACLE], [(r, r)], class_k_gain=1.0)
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
        safe = safe_locomotion_controller_di(cbf_qp, loco, sim_plant, sim_plant.dt, v_max=V_MAX)
    else:
        safe = safe_locomotion_controller(cbf_qp, loco)
    ci = sim_plant.com_indices

    def controller(t, x, v_nom, key, data):  # + goal-reached completion (early stop)
        u, d = safe(t, x, v_nom, key, data)
        reached = jnp.linalg.norm(x[ci[0] : ci[0] + 2] - GOAL) < GOAL_RADIUS
        return u, d._replace(complete=d.complete | reached)

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]

    def nominal(t, x, key, ref):  # proportional to goal on the CoM, saturated
        com = x[ci[0] : ci[0] + 2]
        v = 1.0 * (GOAL - com)
        speed = jnp.linalg.norm(v)
        v = jnp.where(speed > V_MAX, v * (V_MAX / (speed + 1e-9)), v)
        return v, ControllerData()

    return controller, nominal


def main(
    duration=20.0,
    locomotion="policy",
    num_samples=256,
    iterations=2,
    seed=0,
    gif=False,
    view=False,
    robust_bound=None,
    reduced_model="di",
):
    if robust_bound is None:  # measured per model in this scenario, see the module docstring
        robust_bound = {"di": 0.18, "si": 0.16}[reduced_model]
    if TEST_MODE:
        num_samples, iterations = 16, 1
    sim_plant, _loco, x0, pelvis_body, (controller, nominal) = build(
        locomotion, num_samples, iterations, seed, robust_bound, reduced_model
    )
    steps = 5 if TEST_MODE else int(round(duration / sim_plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=sim_plant.dt,
        num_steps=steps,
        plant=sim_plant,
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    wall = time.time() - t0
    states = np.asarray(res["states"])
    ci = sim_plant.com_indices
    com = states[:, ci[0] : ci[0] + 2]
    r = OBSTACLE_RADIUS + ROBOT_RADIUS
    h = (
        ((com[:, 0] - float(OBSTACLE[0])) / r) ** 2
        + ((com[:, 1] - float(OBSTACLE[1])) / r) ** 2
        - 1
    )
    cd = res.controller_data
    v_nom = np.asarray(cd.get("sub_data_v_nom")) if "sub_data_v_nom" in cd else None
    v_safe = np.asarray(cd.get("sub_data_v_safe")) if "sub_data_v_safe" in cd else None
    dist_goal = np.linalg.norm(com - np.asarray(GOAL), axis=1)

    hh = states[:, 2]  # pelvis height
    q = states[:, 3:7]  # pelvis quaternion (w, x, y, z): body z-axis . world z
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    print(f"{steps} steps in {wall:.1f}s")
    print(
        f"h(x) min over run: {h.min():.3f}   (>= 0 means the CoM never entered the keep-out ellipse)"
    )
    print(
        f"closest CoM-obstacle distance: {np.sqrt(((com - np.asarray(OBSTACLE))**2).sum(1)).min():.2f} m (keep-out {r:.2f})"
    )
    print(
        f"distance to goal: start {dist_goal[0]:.2f} -> end {dist_goal[-1]:.2f} (min {dist_goal.min():.2f})"
    )
    print(f"pelvis height min {hh.min():.2f}, upright min {up.min():.2f}")
    reached_at = np.flatnonzero(dist_goal < GOAL_RADIUS)
    print(
        f"goal reached at t={reached_at[0]*sim_plant.dt:.1f}s"
        if reached_at.size
        else "goal not reached within the horizon"
    )
    if v_nom is not None and v_safe is not None:
        print(
            f"CBF active (|v_safe - v_nom| > 1e-3) on {np.mean(np.linalg.norm(v_safe - v_nom, axis=1) > 1e-3)*100:.0f}% of steps"
        )
        # Tracking error of the gait: measured CoM velocity (finite difference of the
        # logged CoM) vs the commanded v_safe. This is the disturbance bound the
        # robust CBF needs (spec milestone 4b) -- measured in the avoidance regime.
        n_live = int(reached_at[0]) if reached_at.size else len(com)
        v_com = np.gradient(com[:n_live], sim_plant.dt, axis=0)
        err = np.linalg.norm(v_com - v_safe[:n_live], axis=1)
        print(
            f"tracking error ||v_com - v_safe||: mean {err.mean():.3f}, p95 {np.percentile(err, 95):.3f}, "
            f"max {err.max():.3f} m/s  (use as --robust bound)"
        )
    if TEST_MODE:
        return float(h.min())
    _TAG[0] = f"{reduced_model}_{'robust' if robust_bound > 0 else 'vanilla'}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _plot(com, h, v_nom, v_safe, hh, sim_plant.dt)
    if gif:
        _render_gif(sim_plant, pelvis_body, states)
    if view:
        _replay_in_viewer(sim_plant, states)
    return float(h.min())


def _plot(com, h, v_nom, v_safe, hh, dt):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(com)) * dt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    r = OBSTACLE_RADIUS + ROBOT_RADIUS
    ax.add_patch(
        plt.Circle(
            np.asarray(OBSTACLE), OBSTACLE_RADIUS, color="crimson", alpha=0.6, label="obstacle"
        )
    )
    ax.add_patch(
        plt.Circle(
            np.asarray(OBSTACLE), r, color="crimson", fill=False, ls="--", label="keep-out (CoM)"
        )
    )
    ax.plot(com[:, 0], com[:, 1], "b-", lw=2, label="CoM path")
    ax.plot(*np.asarray(GOAL), "g*", ms=15, label="goal")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-2, 2)
    ax.legend(loc="upper left")
    ax.set_title("G1 CoM path")
    axes[1].plot(t, h, lw=2)
    axes[1].axhline(0, color="k", ls=":")
    axes[1].set_title("barrier h(x) (>= 0 safe)")
    axes[1].set_xlabel("t [s]")
    if v_nom is not None:
        axes[2].plot(t, v_nom[:, 0], "--", label="v_nom x")
        axes[2].plot(t, v_nom[:, 1], "--", label="v_nom y")
        axes[2].plot(t, v_safe[:, 0], label="v_safe x")
        axes[2].plot(t, v_safe[:, 1], label="v_safe y")
        axes[2].legend()
        axes[2].set_title("commanded CoM velocity")
        axes[2].set_xlabel("t [s]")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"g1_navigate_{_TAG[0]}.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def _add_markers(scn, k=0, t=0.0):
    """Visual-only obstacle / keep-out ring / goal in a mjvScene (renderer or viewer)."""
    import mujoco

    r = OBSTACLE_RADIUS + ROBOT_RADIUS
    cyl, sph = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE
    add_marker(scn, cyl, [OBSTACLE_RADIUS, 0.5, 0], [*OBSTACLE, 0.5], [0.85, 0.15, 0.2, 1.0])
    add_marker(scn, cyl, [r, 0.005, 0], [*OBSTACLE, 0.005], [0.85, 0.15, 0.2, 0.25])
    add_marker(scn, sph, [GOAL_RADIUS, 0, 0], [*GOAL, GOAL_RADIUS], [0.1, 0.8, 0.2, 0.6])


def _render_gif(plant, pelvis_body, states, fps=25):
    path = os.path.join(RESULTS_DIR, f"g1_navigate_{_TAG[0]}.gif")
    render_gif(plant, states, path, track_body=pelvis_body, markers=_add_markers, fps=fps)


def _replay_in_viewer(plant, states):
    replay_in_viewer(plant, states, markers=_add_markers)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument(
        "--robust",
        type=float,
        default=None,
        help="robust CBF with this ||v_com - v_safe|| bound in m/s (default: measured per model, di 0.18 / si 0.16); 0 = vanilla",
    )
    p.add_argument("--locomotion", default="policy", choices=["policy", "mpc"])
    p.add_argument(
        "--reduced-model",
        default="di",
        choices=["si", "di"],
        help="CBF reduced-order model: single integrator (velocity command) or double integrator (acceleration, HOCBF)",
    )
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    if a.view:
        relaunch_under_mjpython_if_needed()
    main(
        a.duration,
        a.locomotion,
        a.samples,
        a.iterations,
        a.seed,
        a.gif,
        a.view,
        a.robust,
        a.reduced_model,
    )
