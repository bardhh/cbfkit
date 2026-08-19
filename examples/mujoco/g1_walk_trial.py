"""Milestone-3 trial harness: can the MJX sampling MPC make the G1 track a planar velocity command?

    python examples/mujoco/g1_walk_trial.py --vx 0.5 --samples 128 --randomizations 1 --duration 4

Prints one JSON line of metrics (mean pelvis velocity, distance, min uprightness/height, fell?) so
trials can be compared in examples/mujoco/G1_WALK_LOG.md. Not a polished example -- see g1_walk.py once
a gait works.
"""

import argparse
import json
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
from cbfkit.systems.mujoco.g1 import G1, friction_randomizer, load_g1, walk_costs
from cbfkit.utils.user_types import ControllerData

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/cbfkit/jax"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run(a) -> dict:
    if a.sim_is_planner:  # diagnostic: remove the planner/sim model mismatch
        sim_plant = MujocoPlant(load_g1())
    else:
        sim_plant = MujocoPlant(load_g1(sim=True), substeps=2)
    mpc_plant = MujocoPlant(load_g1())
    # Per-actuator exploration: legs 0-11 full, waist 12-14 and arms 15-28 scaled down.
    noise = np.full(mpc_plant.nu, a.noise)
    noise[12:15] *= a.waist_noise_scale
    noise[15:] *= a.arm_noise_scale
    g1 = G1(sim_plant.mj_model)
    running, terminal = walk_costs(
        g1,
        target_height=a.height,
        w_velocity=a.w_velocity,
        w_orientation=a.w_orientation,
        w_height=a.w_height,
        w_posture=a.w_posture,
        w_angvel=a.w_angvel,
        w_yaw_rate=a.w_yaw_rate,
        w_control=a.w_control,
        w_balance=a.w_balance,
        w_fall=a.w_fall,
        h_min=a.h_min,
        w_feet=a.w_feet,
        foot_z_max=a.foot_z_max,
        w_qvel=a.w_qvel,
        w_gait=a.w_gait,
        gait_freq=a.gait_freq,
        gait_swing_height=a.gait_swing,
        gait_duty=a.gait_duty,
    )
    mpc = SamplingMpc(
        mpc_plant,
        running,
        terminal,
        num_samples=a.samples,
        plan_horizon=a.horizon,
        noise_level=noise,
        temperature=a.temperature,
        num_knots=a.knots,
        spline_type=a.spline,
        num_randomizations=a.randomizations,
        randomize_model=friction_randomizer(0.5, 2.0) if a.randomizations > 1 else None,
        seed=a.seed,
        update=a.update,
        min_noise_level=a.min_noise,
        iterations=a.iterations,
    )
    v_cmd = jnp.array([a.vx, a.vy])

    def nominal(t, x, key, ref):  # the 2-D command channel: u_nom -> aux of the MPC costs
        return v_cmd, ControllerData()

    x0 = g1.x_stand(sim_plant)
    steps = int(round(a.duration / sim_plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=sim_plant.dt,
        num_steps=steps,
        plant=sim_plant,
        nominal_controller=nominal,
        controller=mpc.as_controller(),
        key=jax.random.PRNGKey(a.seed),
        use_jit=True,
        verbose=False,
    )
    wall = time.time() - t0
    states = np.asarray(res["states"])
    controls = np.asarray(res["controls"])

    from mujoco import mjx

    @jax.jit
    def obs(x):
        d = mjx.kinematics(sim_plant.model, sim_plant.from_state(x))
        return g1.torso_height(d), g1.torso_upright(d)

    h, up = (np.asarray(v) for v in jax.vmap(obs)(jnp.asarray(states)))
    vel = states[:, sim_plant.nq : sim_plant.nq + 2]  # pelvis vx, vy (world)
    pos = states[:, 0:2]
    tail = int(round(2.0 / sim_plant.dt))
    fell = bool((up < 0.5).any() or (h < 0.5).any())
    m = {
        "cmd": [a.vx, a.vy],
        "steps": steps,
        "wall_s": round(wall, 1),
        "mean_v": [round(float(vel[:, 0].mean()), 3), round(float(vel[:, 1].mean()), 3)],
        "mean_v_last2s": [
            round(float(vel[-tail:, 0].mean()), 3),
            round(float(vel[-tail:, 1].mean()), 3),
        ],
        "dist": [round(float(pos[-1, 0] - pos[0, 0]), 2), round(float(pos[-1, 1] - pos[0, 1]), 2)],
        "rms_track_err": round(float(np.sqrt(((vel - np.asarray(v_cmd)) ** 2).sum(1).mean())), 3),
        "min_upright": round(float(up.min()), 2),
        "min_height": round(float(h.min()), 2),
        "fell": fell,
        "yaw_rate_rms": round(float(np.sqrt((states[:, sim_plant.nq + 5] ** 2).mean())), 2),
    }
    if a.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(
            os.path.join(RESULTS_DIR, f"g1_walk_{a.save}.npz"),
            states=states,
            controls=controls,
            meta=json.dumps(vars(a)),
        )
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--randomizations", type=int, default=1)
    p.add_argument("--horizon", type=float, default=0.6)
    p.add_argument("--knots", type=int, default=4)
    p.add_argument("--spline", default="zero")
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--height", type=float, default=0.9)
    p.add_argument("--w-velocity", type=float, default=10.0)
    p.add_argument("--w-orientation", type=float, default=10.0)
    p.add_argument("--w-height", type=float, default=5.0)
    p.add_argument("--w-posture", type=float, default=0.1)
    p.add_argument("--w-angvel", type=float, default=0.1)
    p.add_argument("--w-yaw-rate", type=float, default=1.0)
    p.add_argument("--w-control", type=float, default=0.0)
    p.add_argument("--w-balance", type=float, default=0.0)
    p.add_argument("--w-fall", type=float, default=0.0)
    p.add_argument("--update", default="mppi", choices=["mppi", "cma"])
    p.add_argument("--min-noise", type=float, default=None)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--h-min", type=float, default=0.75)
    p.add_argument("--w-feet", type=float, default=0.0)
    p.add_argument("--foot-z-max", type=float, default=0.15)
    p.add_argument("--w-qvel", type=float, default=0.0)
    p.add_argument("--w-gait", type=float, default=0.0)
    p.add_argument("--gait-freq", type=float, default=1.5)
    p.add_argument("--gait-swing", type=float, default=0.08)
    p.add_argument("--gait-duty", type=float, default=0.5)
    p.add_argument("--arm-noise-scale", type=float, default=1.0)
    p.add_argument("--waist-noise-scale", type=float, default=1.0)
    p.add_argument(
        "--sim-is-planner", action="store_true", help="use the planner model as the sim model too"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--save", default="", help="tag: save states/controls to results/g1_walk_<tag>.npz"
    )
    a = p.parse_args()
    print(json.dumps({"args": {k: v for k, v in vars(a).items() if k != "save"}}))
    print(json.dumps(run(a)))
