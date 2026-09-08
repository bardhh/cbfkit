"""Unitree G1 humanoid standup with an MJX sampling MPC, driven by ``sim.execute(plant=...)``.

Replicates hydrax's ``humanoid_standup`` example on CBFKit's simulator: the G1
starts lying on its side and the sampling MPC (MPPI over spline knots, friction
domain randomisation) drives its 29 position-servo targets to bring it upright.

Requires the optional extra:  pip install cbfkit[mujoco]
On first run the 51 G1 meshes (~34 MB) are downloaded from the MuJoCo Menagerie
(pinned commit, SHA-256 verified) into ~/.cache/cbfkit/ (or $CBFKIT_ASSET_DIR).
Model license: BSD-3-Clause, Unitree Robotics (see cbfkit/systems/mujoco/models/g1/).

Usage
-----
    python examples/mujoco/g1_standup.py                # run + save plots (~2-4 min on CPU)
    python examples/mujoco/g1_standup.py --gif          # also render an animation (offscreen)
    python examples/mujoco/g1_standup.py --view         # replay in the MuJoCo viewer (macOS: auto-relaunches under mjpython)
    python examples/mujoco/g1_standup.py --duration 3 --samples 64 --randomizations 1   # faster
    CBFKIT_TEST_MODE=1 python examples/mujoco/g1_standup.py   # short smoke run, no plots

Two plants, as in hydrax: a *simulation* plant with stiffer contact and a finer
step (timestep 0.01, 2 substeps -> 50 Hz control) and a *planner* plant
(timestep 0.02) that the MPC rolls out. Both share the same flat state layout.
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
from cbfkit.controllers.mjx_sampling_mpc import SamplingMpc
from cbfkit.systems.mujoco import MujocoPlant
from cbfkit.systems.mujoco.g1 import G1, friction_randomizer, load_g1, standup_costs
from cbfkit.systems.mujoco.viewer_utils import relaunch_under_mjpython_if_needed, under_mjpython

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Persistent XLA compile cache: the G1 closed loop takes ~10-20 s to compile.
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/cbfkit/jax"))


def build(num_samples: int, num_randomizations: int, seed: int = 0):
    """Simulation plant, planner plant, G1 ids, and the sampling MPC (hydrax's standup config)."""
    sim_plant = MujocoPlant(load_g1(sim=True), substeps=2)  # dt = 0.02
    mpc_plant = MujocoPlant(load_g1())  # planner model, dt = 0.02
    assert abs(sim_plant.dt - mpc_plant.dt) < 1e-12
    g1 = G1(sim_plant.mj_model)
    running, terminal = standup_costs(g1, target_height=0.9)
    mpc = SamplingMpc(
        mpc_plant,
        running,
        terminal,
        num_samples=num_samples,
        plan_horizon=0.6,
        noise_level=0.3,
        temperature=0.1,
        num_knots=4,
        spline_type="zero",
        num_randomizations=num_randomizations,
        randomize_model=friction_randomizer(0.5, 2.0) if num_randomizations > 1 else None,
        seed=seed,
    )
    return sim_plant, mpc_plant, g1, mpc


def main(
    duration: float = 4.0,
    num_samples: int = 128,
    num_randomizations: int = 4,
    seed: int = 0,
    gif: bool = False,
    view: bool = False,
) -> float:
    """Run the standup. Returns the mean torso height over the final second."""
    if TEST_MODE:
        num_samples, num_randomizations = 16, 1
    sim_plant, _, g1, mpc = build(num_samples, num_randomizations, seed)
    controller = mpc.as_controller()

    num_steps = 5 if TEST_MODE else int(round(duration / sim_plant.dt))
    x0 = g1.x_fallen(sim_plant)

    print(
        f"G1: nq={sim_plant.nq} nv={sim_plant.nv} nu={sim_plant.nu} | {num_samples} samples x "
        f"{num_randomizations} randomizations, horizon 0.6 s, control {1 / sim_plant.dt:.0f} Hz, "
        f"{num_steps} steps"
    )
    t0 = time.time()
    results = sim.execute(
        x0=x0,
        dt=sim_plant.dt,
        num_steps=num_steps,
        plant=sim_plant,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    elapsed = time.time() - t0

    states = np.asarray(results["states"])
    controls = np.asarray(results["controls"])
    t = np.arange(num_steps) * sim_plant.dt

    # Observables from the logged flat states (torso site needs kinematics).
    heights, uprights = _observables(sim_plant, g1, states)
    tail = max(1, int(round(1.0 / sim_plant.dt)))
    score = float(np.mean(heights[-tail:]))
    print(f"{num_steps} steps in {elapsed:.1f}s (incl. JIT compile)")
    print(
        f"torso height: start {heights[0]:.2f} m -> final-second mean {score:.2f} m (target 0.90)"
    )
    print(
        f"torso upright (cos): start {uprights[0]:.2f} -> final-second mean {np.mean(uprights[-tail:]):.2f}"
    )
    if TEST_MODE:
        return score

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _plot(t, heights, uprights, controls)
    if gif:
        _render_gif(sim_plant, g1, states, controls)
    if view:
        _replay_in_viewer(sim_plant, states, controls)
    return score


def _observables(plant, g1, states):
    """Torso height and uprightness per logged step, via jitted forward kinematics."""
    from mujoco import mjx

    @jax.jit
    def obs(x):
        d = mjx.kinematics(plant.model, plant.from_state(x))
        return g1.torso_height(d), g1.torso_upright(d)

    h, u = jax.vmap(obs)(jnp.asarray(states))
    return np.asarray(h), np.asarray(u)


def _plot(t, heights, uprights, controls):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(t, heights, lw=2)
    axes[0].axhline(0.9, color="k", ls=":", lw=1, label="target 0.9 m")
    axes[0].set_ylabel("torso height [m]")
    axes[0].legend(loc="lower right")
    axes[1].plot(t, uprights, lw=2)
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    axes[1].set_ylabel("torso upright (cos)")
    axes[1].set_ylim(-1.05, 1.05)
    axes[2].plot(t, np.linalg.norm(np.diff(controls, axis=0, prepend=controls[:1]), axis=1), lw=1)
    axes[2].set_ylabel("||Δ joint targets|| per step")
    axes[2].set_xlabel("time [s]")
    axes[0].set_title("Unitree G1 standup: MJX sampling MPC through cbfkit execute(plant=...)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "g1_standup.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def _replay(plant, states, controls):
    import mujoco

    m = plant.mj_model
    d = mujoco.MjData(m)
    for k in range(states.shape[0]):
        d.qpos[:] = states[k, : plant.nq]
        d.qvel[:] = states[k, plant.nq : plant.nq + plant.nv]
        d.ctrl[:] = controls[k]
        mujoco.mj_forward(m, d)
        yield d, k


def _camera(plant, g1):
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = g1.pelvis_body
    cam.distance = 3.0
    cam.azimuth = 135
    cam.elevation = -15
    return cam


def _render_gif(plant, g1, states, controls, fps: int = 25):
    import matplotlib
    import mujoco

    matplotlib.use("Agg")
    from matplotlib import animation
    from matplotlib import pyplot as plt

    try:
        renderer = mujoco.Renderer(plant.mj_model, height=360, width=640)
    except Exception as exc:  # noqa: BLE001 -- offscreen GL is platform-dependent
        print(f"offscreen rendering unavailable ({exc}); skipping GIF")
        return
    every = max(1, int(round(1.0 / (fps * plant.dt))))
    cam = _camera(plant, g1)
    frames = []
    for d, k in _replay(plant, states, controls):
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
    path = os.path.join(RESULTS_DIR, "g1_standup.gif")
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {path}")


def _replay_in_viewer(plant, states, controls):
    import mujoco

    if not under_mjpython():
        print("viewer skipped: on macOS it needs `mjpython` (rerun with --view; it relaunches).")
        return
    import mujoco.viewer

    m = plant.mj_model
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            for dd, _ in _replay(plant, states, controls):
                if not viewer.is_running():
                    break
                d.qpos[:] = dd.qpos
                d.qvel[:] = dd.qvel
                mujoco.mj_forward(m, d)
                viewer.sync()
                time.sleep(plant.dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duration", type=float, default=4.0, help="simulated seconds (default 4)")
    parser.add_argument("--samples", type=int, default=128, help="MPPI samples (default 128)")
    parser.add_argument(
        "--randomizations", type=int, default=4, help="friction DR count (default 4)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gif", action="store_true", help="render an offscreen animation to results/"
    )
    parser.add_argument("--view", action="store_true", help="replay the run in the MuJoCo viewer")
    args = parser.parse_args()
    if args.view:
        relaunch_under_mjpython_if_needed()
    main(
        duration=args.duration,
        num_samples=args.samples,
        num_randomizations=args.randomizations,
        seed=args.seed,
        gif=args.gif,
        view=args.view,
    )
