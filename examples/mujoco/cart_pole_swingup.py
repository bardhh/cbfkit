"""Cart-pole swing-up with an MJX sampling MPC, driven by ``sim.execute(plant=...)``.

This is the smallest end-to-end use of the MuJoCo plant backend:

    MujocoPlant(model)  ->  execute(plant=..., controller=SamplingMpc(...).as_controller())

Requires the optional extra:  pip install cbfkit[mujoco]

Usage
-----
    python examples/mujoco/cart_pole_swingup.py               # run + save plots
    python examples/mujoco/cart_pole_swingup.py --gif         # also render an animation (offscreen)
    python examples/mujoco/cart_pole_swingup.py --view        # replay in the MuJoCo viewer
    CBFKIT_TEST_MODE=1 python examples/mujoco/cart_pole_swingup.py   # short run, no plots

On macOS the interactive viewer needs MuJoCo's ``mjpython`` launcher; with
``--view`` the script re-execs itself under the venv's ``mjpython`` automatically
(set ``CBFKIT_NO_MJPYTHON=1`` to disable). Running ``mjpython <script> --view``
directly works too.
"""

import argparse
import os
import sys
import time

# Add the project root to the path so we can import cbfkit + examples.
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax
import jax.numpy as jnp
import numpy as np

import cbfkit.simulation.simulator as sim
from cbfkit.controllers.mjx_sampling_mpc import SamplingMpc
from cbfkit.systems.mujoco import MujocoPlant, load_model
from cbfkit.systems.mujoco.viewer_utils import relaunch_under_mjpython_if_needed, under_mjpython

# In test mode we shorten the run and skip plots/rendering.
TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# Task: costs are functions of the MJX state, exactly as in hydrax. ``aux`` is
# the nominal control forwarded by the simulator (unused here).
# ---------------------------------------------------------------------------
def upright_distance(qpos):
    """0 when the pole is upright, 4 when hanging straight down."""
    theta = qpos[1] + jnp.pi  # theta = 0 is "down" in this model
    return (jnp.cos(theta) - 1.0) ** 2 + jnp.sin(theta) ** 2


def running_cost(data, u, aux):
    return (
        upright_distance(data.qpos)
        + data.qpos[0] ** 2  # keep the cart centred
        + 0.01 * jnp.sum(data.qvel**2)
        + 0.01 * jnp.sum(u**2)
    )


def terminal_cost(data, aux):
    return 10.0 * upright_distance(data.qpos) + data.qpos[0] ** 2 + 0.01 * jnp.sum(data.qvel**2)


def build(num_samples: int = 128):
    """Plant + controller. Build once and reuse: their identity keys the JIT cache."""
    plant = MujocoPlant(load_model("cart_pole"), substeps=2)  # model dt 0.01 -> control at 50 Hz
    mpc = SamplingMpc(
        plant,
        running_cost,
        terminal_cost,
        num_samples=num_samples,
        plan_horizon=1.0,
        noise_level=0.3,
        temperature=0.1,
        num_knots=4,
        spline_type="linear",
    )
    return plant, mpc


def main(duration: float = 4.0, seed: int = 0, gif: bool = False, view: bool = False) -> float:
    """Run the swing-up. Returns the mean upright-distance over the final second."""
    plant, mpc = build(num_samples=32 if TEST_MODE else 128)
    controller = mpc.as_controller()

    num_steps = 10 if TEST_MODE else int(round(duration / plant.dt))
    x0 = jnp.zeros(plant.state_dim)  # [qpos | qvel | com_xyz], pole hanging down

    t0 = time.time()
    results = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=num_steps,
        plant=plant,  # <- replaces dynamics= + integrator=
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,  # the JIT path is the supported plant path
        verbose=not TEST_MODE,
    )
    elapsed = time.time() - t0

    states = np.asarray(results["states"])  # (num_steps, plant.state_dim)
    controls = np.asarray(results["controls"])  # (num_steps, plant.nu)
    t = np.arange(num_steps) * plant.dt
    dist = np.asarray([float(upright_distance(jnp.asarray(q))) for q in states[:, :2]])
    tail = max(1, int(round(1.0 / plant.dt)))
    score = float(dist[-tail:].mean())

    reached = np.flatnonzero(dist < 0.5)
    print(f"{num_steps} steps in {elapsed:.1f}s (incl. JIT compile)")
    print(
        f"upright within 30 deg first at t={t[reached[0]]:.2f}s"
        if reached.size
        else "never upright"
    )
    print(f"mean upright-distance over final second: {score:.4f}   (0 = balanced, 4 = hanging)")
    print(f"cart travel: |x| <= {np.abs(states[:, 0]).max():.2f} m   (rail limit 1.8 m)")

    if TEST_MODE:
        return score

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _plot(t, states, controls, dist, plant)
    if gif:
        _render_gif(plant, states, controls)
    if view:
        _replay_in_viewer(plant, states, controls)
    return score


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _plot(t, states, controls, dist, plant):
    import matplotlib

    # Figures are only saved, never shown. Agg also keeps matplotlib off the GUI
    # thread, which matters under mjpython (Cocoa owns the main thread there).
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    # theta = 0 is hanging down, so the angle from upright is wrap(theta - pi).
    axes[0].plot(t, np.rad2deg(np.mod(states[:, 1], 2 * np.pi) - np.pi), lw=2)
    axes[0].axhline(0, color="k", ls=":", lw=1)
    axes[0].set_ylabel("pole angle from upright [deg]")
    axes[1].plot(t, states[:, 0], lw=2, label="cart x")
    axes[1].plot(t, states[:, plant.com_indices[0]], lw=1.5, ls="--", label="CoM x")
    axes[1].axhline(1.8, color="r", ls=":", lw=1)
    axes[1].axhline(-1.8, color="r", ls=":", lw=1)
    axes[1].set_ylabel("position [m]")
    axes[1].legend(loc="upper right")
    axes[2].step(t, controls[:, 0], where="post", lw=1.5)
    axes[2].set_ylabel("ctrl (motor, ±1)")
    axes[2].set_xlabel("time [s]")
    axes[0].set_title("Cart-pole swing-up: MJX sampling MPC through cbfkit execute(plant=...)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "cart_pole_swingup.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def _replay(plant, states, controls):
    """Yield (mj_data, k) after replaying step k of the logged run on the CPU model."""
    import mujoco

    m = plant.mj_model
    d = mujoco.MjData(m)
    for k in range(states.shape[0]):
        d.qpos[:] = states[k, : plant.nq]
        d.qvel[:] = states[k, plant.nq : plant.nq + plant.nv]
        d.ctrl[:] = controls[k]
        mujoco.mj_forward(m, d)
        yield d, k


def _render_gif(plant, states, controls, fps: int = 25):
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
    frames = []
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(plant.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "lookatcart")
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
    path = os.path.join(RESULTS_DIR, "cart_pole_swingup.gif")
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {path}")


def _replay_in_viewer(plant, states, controls):
    import mujoco

    if not under_mjpython():
        print(
            "viewer skipped: on macOS it needs `mjpython` "
            f"(run `{sys.executable.replace('python', 'mjpython', 1)} {sys.argv[0]} --view`)."
        )
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gif", action="store_true", help="render an offscreen animation to results/"
    )
    parser.add_argument("--view", action="store_true", help="replay the run in the MuJoCo viewer")
    args = parser.parse_args()
    if args.view:
        # macOS: the passive viewer needs mjpython. Re-exec now, before the JIT
        # work, so `python ... --view` and `mjpython ... --view` behave the same.
        relaunch_under_mjpython_if_needed()
    main(duration=args.duration, seed=args.seed, gif=args.gif, view=args.view)
