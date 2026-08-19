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
  mpc               the in-repo MJX sampling MPC on the 29-DoF model (G1_WALK_LOG.md): forward
                    locomotion with stumbles; kept for comparison.

The CBF certifies the *commanded* CoM velocity; the gait's tracking error is what the robust
variant's disturbance bound must cover (spec milestone 4b).

    python examples/mujoco/g1_navigate.py [--locomotion policy|mpc] [--gif] [--view] [--duration 12]
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
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.controllers.mjx_sampling_mpc import SamplingMpc
from cbfkit.systems.mujoco import MujocoPlant
from cbfkit.systems.mujoco.g1 import G1, load_g1, walk_costs
from cbfkit.systems.mujoco.unitree_policy import (
    UnitreeG1WalkPolicy,
    make_g1_12dof_plant,
    x0_standing,
)
from cbfkit.systems.mujoco.reduced_order import (
    com_obstacle_barriers,
    embedded_single_integrator,
    safe_locomotion_controller,
)
from cbfkit.systems.mujoco.viewer_utils import relaunch_under_mjpython_if_needed, under_mjpython
from cbfkit.utils.user_types import ControllerData

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/cbfkit/jax"))

GOAL = jnp.array([4.0, 0.0])
OBSTACLE = jnp.array([2.0, 0.0])  # centre, on the straight line to the goal
OBSTACLE_RADIUS = 0.35
ROBOT_RADIUS = 0.35  # planar inflation so "CoM outside the ellipse" => "body clear"
V_MAX = 0.5


def build(locomotion: str = "policy", num_samples: int = 256, iterations: int = 2, seed: int = 0):
    if locomotion == "policy":
        sim_plant = make_g1_12dof_plant()  # 12-DoF legs, PD at 500 Hz, dt 0.02
        loco = UnitreeG1WalkPolicy().as_controller()
        x0 = x0_standing(sim_plant)
        pelvis_body = int(sim_plant.mj_model.body("pelvis").id)
        return sim_plant, loco, x0, pelvis_body, _finish(sim_plant, loco)

    sim_plant = MujocoPlant(load_g1(sim=True), substeps=2)
    mpc_plant = MujocoPlant(load_g1())
    g1 = G1(sim_plant.mj_model)

    # Locomotion: best-so-far walk configuration (G1_WALK_LOG.md, t14/t20 class).
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
        _finish(sim_plant, mpc.as_controller()),
    )


def _finish(sim_plant, loco):
    """Safety layer + nominal, identical for every locomotion layer."""
    r = OBSTACLE_RADIUS + ROBOT_RADIUS
    dyn = embedded_single_integrator(sim_plant.state_dim, sim_plant.com_indices)
    barriers = com_obstacle_barriers(sim_plant, [OBSTACLE], [(r, r)], class_k_gain=1.0)
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([V_MAX, V_MAX]), dynamics_func=dyn, barriers=barriers
    )
    controller = safe_locomotion_controller(cbf_qp, loco)
    ci = sim_plant.com_indices

    def nominal(t, x, key, ref):  # proportional to goal on the CoM, saturated
        com = x[ci[0] : ci[0] + 2]
        v = 1.0 * (GOAL - com)
        speed = jnp.linalg.norm(v)
        v = jnp.where(speed > V_MAX, v * (V_MAX / (speed + 1e-9)), v)
        return v, ControllerData()

    return controller, nominal


def main(
    duration=12.0, locomotion="policy", num_samples=256, iterations=2, seed=0, gif=False, view=False
):
    if TEST_MODE:
        num_samples, iterations = 16, 1
    sim_plant, _loco, x0, pelvis_body, (controller, nominal) = build(
        locomotion, num_samples, iterations, seed
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
    if v_nom is not None and v_safe is not None:
        print(
            f"CBF active (|v_safe - v_nom| > 1e-3) on {np.mean(np.linalg.norm(v_safe - v_nom, axis=1) > 1e-3)*100:.0f}% of steps"
        )
    if TEST_MODE:
        return float(h.min())
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
    path = os.path.join(RESULTS_DIR, "g1_navigate.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def _replay(plant, states):
    import mujoco

    m = plant.mj_model
    d = mujoco.MjData(m)
    for k in range(states.shape[0]):
        d.qpos[:] = states[k, : plant.nq]
        d.qvel[:] = states[k, plant.nq : plant.nq + plant.nv]
        mujoco.mj_forward(m, d)
        yield d, k


def _render_gif(plant, pelvis_body, states, fps=25):
    import matplotlib
    import mujoco

    matplotlib.use("Agg")
    from matplotlib import animation, pyplot as plt

    try:
        renderer = mujoco.Renderer(plant.mj_model, height=360, width=640)
    except Exception as exc:  # noqa: BLE001
        print(f"offscreen rendering unavailable ({exc}); skipping GIF")
        return
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = pelvis_body
    cam.distance = 4.0
    cam.azimuth = 160
    cam.elevation = -20
    every = max(1, int(round(1.0 / (fps * plant.dt))))
    frames = []
    for d, k in _replay(plant, states):
        if k % every:
            continue
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render().copy())
    fig = plt.figure(figsize=(6.4, 3.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(frames[0])
    anim = animation.FuncAnimation(
        fig, lambda i: (im.set_data(frames[i]),), frames=len(frames), interval=1000 / fps
    )
    path = os.path.join(RESULTS_DIR, "g1_navigate.gif")
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {path}")


def _replay_in_viewer(plant, states):
    import mujoco

    if not under_mjpython():
        print("viewer skipped: needs mjpython (rerun with --view; it relaunches)")
        return
    import mujoco.viewer

    m = plant.mj_model
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            for dd, _ in _replay(plant, states):
                if not viewer.is_running():
                    break
                d.qpos[:] = dd.qpos
                d.qvel[:] = dd.qvel
                mujoco.mj_forward(m, d)
                viewer.sync()
                time.sleep(plant.dt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--locomotion", default="policy", choices=["policy", "mpc"])
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    if a.view:
        relaunch_under_mjpython_if_needed()
    main(a.duration, a.locomotion, a.samples, a.iterations, a.seed, a.gif, a.view)
