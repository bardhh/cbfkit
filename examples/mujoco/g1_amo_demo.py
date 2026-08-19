"""Unitree G1 walking with AMO's whole-body policy: torso yaw / roll / pitch / height
commanded *while walking* -- the agility the 12-DoF walking policy cannot do.

The robot walks a straight line at ``VX`` m/s through a schedule of torso commands
(upright, look left/right, lean left/right, duck, bow -- each held ``PHASE`` seconds),
using ``cbfkit.systems.mujoco.amo_policy`` (UCSD's AMO, RSS 2025, ported to JAX and run
on the MJX 23-DoF G1). Reported per phase: commanded vs realised waist joint angles,
CoM height, and forward-speed tracking -- the numbers that matter when the CBF layer
later uses torso lean to shrink the robot's swept width through a gap.

Measured (MJX, CPU, 40 s, vx = 0.4 m/s; waist joint = the commanded DoF, the rest of the
torso motion comes from hip lean):

    phase        cmd            realised (waist / com)         speed
    look left    yaw  +1.20     +1.05                          0.32
    look right   yaw  -1.20     -1.17                          0.21
    lean left    roll +0.50     +0.35                          0.27
    lean right   roll -0.50     -0.33                          0.28
    duck         z 0.75->0.40   com z 0.67->0.48               0.26
    bow          pitch +0.80    +0.56                          0.29

Upright min 0.97 -- it never stumbles through any of it. Forward speed realises ~0.3 of
the commanded 0.4 m/s (an MJX sim2sim gap: AMO was trained in IsaacGym and demoed by its
authors in MuJoCo-CPU; the gait transfers, the speed calibration does not fully).

    python examples/mujoco/g1_amo_demo.py [--duration T] [--vx V] [--gif] [--view]
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
from cbfkit.systems.mujoco.amo_policy import (
    AmoWholeBodyPolicy,
    make_g1_23dof_plant,
    x0_standing,
)
from cbfkit.utils.user_types import ControllerData, PlannerData

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

VX = 0.4  # m/s forward command (AMO I.D. range +-0.5)
PHASE = 5.0  # s per torso command
# (height_delta, torso_yaw, torso_pitch, torso_roll) -- all within the I.D. ranges
SCHEDULE = [
    ("upright", (0.0, 0.0, 0.0, 0.0)),
    ("look left", (0.0, 1.2, 0.0, 0.0)),
    ("look right", (0.0, -1.2, 0.0, 0.0)),
    ("lean left", (0.0, 0.0, 0.0, 0.5)),
    ("lean right", (0.0, 0.0, 0.0, -0.5)),
    ("duck", (-0.35, 0.0, 0.0, 0.0)),
    ("bow", (0.0, 0.0, 0.8, 0.0)),
    ("upright", (0.0, 0.0, 0.0, 0.0)),
]
DEFAULT_DURATION = PHASE * len(SCHEDULE)


def torso_schedule(t):
    """Piecewise-constant torso command from ``SCHEDULE`` (JIT-safe: gather by index)."""
    table = jnp.asarray([c for _, c in SCHEDULE], dtype=float)
    idx = jnp.clip(jnp.floor(t / PHASE).astype(int), 0, len(SCHEDULE) - 1)
    return table[idx]


def build():
    plant = make_g1_23dof_plant()
    policy = AmoWholeBodyPolicy()
    controller = policy.as_controller(torso_command=torso_schedule)
    x0 = x0_standing(plant)
    torso_body = plant.body_id("torso_link")
    pelvis_body = plant.body_id("pelvis")

    def nominal(t, x, key, ref):
        return jnp.array([VX, 0.0]), ControllerData()

    return plant, x0, pelvis_body, torso_body, nominal, controller


def _rpy_np(quat):
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.stack([roll, pitch, yaw], axis=-1)


def main(duration=DEFAULT_DURATION, seed=0, gif=False, view=False, vx=VX):
    global VX
    VX = vx
    plant, x0, pelvis_body, torso_body, nominal, controller = build()
    steps = 5 if TEST_MODE else int(round(duration / plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        plant=plant,
        planner_data=PlannerData.from_constant(jnp.array([1e3, 0.0])),
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    wall = time.time() - t0
    S = np.asarray(res["states"])
    cd = res.controller_data
    cmd = np.asarray(cd["sub_data_amo_cmd"])  # (T, 7)
    t = np.arange(len(S)) * plant.dt
    pelvis_rpy = _rpy_np(S[:, 3:7])
    com = S[:, plant.com_indices[0] : plant.com_indices[0] + 2]
    com_z = S[:, plant.com_indices[0] + 2]
    # torso posture relative to the pelvis: the commanded quantity is the waist, read joints
    waist = S[:, 19:22]  # waist yaw, roll, pitch qpos
    v_com = np.gradient(com, plant.dt, axis=0) if len(S) > 1 else np.zeros_like(com)
    speed = np.linalg.norm(v_com, axis=1)
    up = 1 - 2 * (S[:, 4] ** 2 + S[:, 5] ** 2)

    print(f"{steps} steps in {wall:.1f}s")
    print(f"pelvis height min {S[:, 2].min():.2f}, upright min {up.min():.2f}")
    print(
        f"forward speed: commanded {vx:.2f}, realised mean {speed.mean():.2f} "
        f"(p95 |err| {np.percentile(np.abs(speed - vx), 95):.2f}) m/s"
    )
    if not TEST_MODE:
        print(
            f"{'phase':<12}{'cmd yaw':>9}{'waist yaw':>10}{'cmd roll':>10}{'waist roll':>11}"
            f"{'cmd pitch':>10}{'waist pitch':>12}{'cmd z':>7}{'com z':>7}{'speed':>7}"
        )
        for i, (name, (dh, ty, tp, tr)) in enumerate(SCHEDULE):
            m = (t >= i * PHASE + 2.0) & (t < (i + 1) * PHASE)  # settle 2 s into the phase
            if not m.any():
                continue
            print(
                f"{name:<12}{ty:>9.2f}{waist[m, 0].mean():>10.2f}{tr:>10.2f}{waist[m, 1].mean():>11.2f}"
                f"{tp:>10.2f}{waist[m, 2].mean():>12.2f}{0.75 + dh:>7.2f}{com_z[m].mean():>7.2f}"
                f"{speed[m].mean():>7.2f}"
            )
    h_min = float(S[:, 2].min())
    if TEST_MODE:
        print(f"torso height min {h_min:.2f}")  # smoke-test marker
        return h_min
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _plot(t, cmd, waist, com_z, speed, pelvis_rpy)
    if gif or view:
        from cbfkit.systems.mujoco.viewer_utils import render_gif, replay_in_viewer

        if gif:
            render_gif(
                plant,
                S,
                os.path.join(RESULTS_DIR, "g1_amo_demo.gif"),
                track_body=pelvis_body,
                distance=3.5,
                elevation=-15.0,
            )
        if view:
            replay_in_viewer(plant, S)
    return h_min


def _plot(t, cmd, waist, com_z, speed, pelvis_rpy):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (ci, wi, name) in zip(
        axes.flat, [(4, 0, "torso yaw"), (6, 1, "torso roll"), (5, 2, "torso pitch")]
    ):
        ax.plot(t, cmd[:, ci], "--", label="command")
        ax.plot(t, waist[:, wi], label="waist joint")
        ax.set_title(f"{name} [rad]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax = axes.flat[3]
    ax.plot(t, cmd[:, 3] + 0.75, "--", label="height command")
    ax.plot(t, com_z, label="CoM z")
    ax2 = ax.twinx()
    ax2.plot(t, speed, color="tab:green", lw=0.8, alpha=0.7, label="|v_com|")
    ax.set_title("height [m] / speed [m/s]")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    for a in axes.flat:
        a.set_xlabel("t [s]")
    for i in range(1, len(SCHEDULE)):
        for a in axes.flat:
            a.axvline(i * PHASE, color="gray", lw=0.5, ls=":")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "g1_amo_demo.png")
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    p.add_argument("--vx", type=float, default=VX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    if a.view:
        from cbfkit.systems.mujoco.viewer_utils import relaunch_under_mjpython_if_needed

        relaunch_under_mjpython_if_needed()
    main(a.duration, a.seed, a.gif, a.view, a.vx)
