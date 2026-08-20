"""Gait comparison of the three ported G1 policies: Unitree (12-DoF), UCSD AMO (23-DoF),
NVIDIA GR00T GEAR-WBC (29-DoF) -- same command, same simulator, measured side by side.

Each policy walks straight at ``VX`` for ``DURATION`` seconds in MJX on its own plant.
"Natural" gets numbers: realised forward speed (command tracking), CoM height bounce
(std), base wobble (roll/pitch std and gyro RMS), and command smoothness (PD-target
rate); plus a GIF per policy (``--gif``) for the eyeball test.

Measured (MJX, CPU, vx = 0.4, 12 s, settle 3 s):

    policy    v_real  v_err_p95  com-z std  roll std  pitch std  gyro RMS  target-rate
    unitree     0.38       0.05       4 mm      2.3        0.3       0.36          7.8
    amo         0.24       0.17       5 mm      0.4        0.7       0.49          5.7
    groot       0.31       0.10       2 mm      3.0        0.3       0.52          6.7

Reading: GR00T tracks the command better than AMO and rides the flattest (2 mm CoM
bounce); its 3.0 deg roll std is a visible side-to-side weight shift (judge from the
GIF whether it reads as human-like or as sway); its arms hang naturally (zero pose)
where the Unitree policy locks a bent-arm default. The Unitree 12-DoF policy remains
the best pure command tracker. GIFs: ``results/g1_walk_{unitree,amo,groot}.gif``.

    python examples/mujoco/g1_walk_compare.py [--vx V] [--duration T] [--gif]
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
from cbfkit.utils.user_types import ControllerData, PlannerData

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

VX = 0.4
DURATION = 2.0 if TEST_MODE else 12.0
SETTLE = 0.5 if TEST_MODE else 3.0


def build(which):
    if which == "unitree":
        from cbfkit.systems.mujoco.unitree_policy import (
            UnitreeG1WalkPolicy,
            make_g1_12dof_plant,
            x0_standing,
        )

        plant = make_g1_12dof_plant()
        return plant, UnitreeG1WalkPolicy().as_controller(), x0_standing(plant)
    if which == "amo":
        from cbfkit.systems.mujoco import amo_policy as amo

        plant = amo.make_g1_23dof_plant()
        return plant, amo.AmoWholeBodyPolicy().as_controller(), amo.x0_standing(plant)
    if which == "groot":
        from cbfkit.systems.mujoco import groot_policy as groot

        plant = groot.make_g1_29dof_plant()
        return plant, groot.GrootGearWbcPolicy().as_controller(), groot.x0_standing(plant)
    raise ValueError(which)


def run_one(which, vx, duration, gif=False):
    plant, ctrl, x0 = build(which)

    def nominal(t, x, key, ref):
        return jnp.array([vx, 0.0]), ControllerData()

    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=int(round(duration / plant.dt)),
        plant=plant,
        planner_data=PlannerData.from_constant(jnp.array([1e3, 0.0])),
        nominal_controller=nominal,
        controller=ctrl,
        key=jax.random.PRNGKey(0),
        use_jit=True,
        verbose=False,
    )
    wall = time.time() - t0
    S = np.asarray(res["states"])
    n0 = int(round(SETTLE / plant.dt))
    ci = plant.com_indices
    com = S[:, ci[0] : ci[0] + 2]
    com_z = S[n0:, ci[0] + 2]
    v_com = np.gradient(com, plant.dt, axis=0)[n0:]
    qw, qx, qy, qz = S[n0:, 3], S[n0:, 4], S[n0:, 5], S[n0:, 6]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    omega = S[n0:, plant.nq + 3 : plant.nq + 6]
    U = np.asarray(res["controls"])[n0:]
    dU = np.diff(U, axis=0) / plant.dt
    m = {
        "policy": which,
        "v_realised": float(v_com[:, 0].mean()),
        "v_err_p95": float(np.percentile(np.abs(v_com[:, 0] - vx), 95)),
        "lateral_drift": float(abs(com[-1, 1] - com[n0, 1]) / max(duration - SETTLE, 1e-9)),
        "com_z_std_mm": float(com_z.std() * 1e3),
        "roll_std_deg": float(np.rad2deg(roll.std())),
        "pitch_std_deg": float(np.rad2deg(pitch.std())),
        "gyro_rms": float(np.sqrt((omega**2).sum(1).mean())),
        "target_rate_rms": float(np.sqrt((dU**2).sum(1).mean())),
        "upright_min": float((1 - 2 * (qx**2 + qy**2)).min()),
        "wall_s": wall,
    }
    if gif and not TEST_MODE:
        from cbfkit.systems.mujoco.viewer_utils import render_gif

        os.makedirs(RESULTS_DIR, exist_ok=True)
        render_gif(
            plant,
            S,
            os.path.join(RESULTS_DIR, f"g1_walk_{which}.gif"),
            track_body=int(plant.mj_model.body("pelvis").id),
            distance=3.0,
            elevation=-15.0,
        )
    return m


def main(vx=VX, duration=DURATION, gif=False):
    rows = []
    for which in ("unitree", "amo", "groot"):
        m = run_one(which, vx, duration, gif)
        rows.append(m)
        print(f"[{which}] done in {m['wall_s']:.0f}s")
    cols = [
        ("policy", "{}"),
        ("v_realised", "{:.2f}"),
        ("v_err_p95", "{:.2f}"),
        ("lateral_drift", "{:.3f}"),
        ("com_z_std_mm", "{:.0f}"),
        ("roll_std_deg", "{:.1f}"),
        ("pitch_std_deg", "{:.1f}"),
        ("gyro_rms", "{:.2f}"),
        ("target_rate_rms", "{:.1f}"),
        ("upright_min", "{:.2f}"),
    ]
    print("  ".join(f"{k:>15s}" for k, _ in cols))
    for r in rows:
        print("  ".join(f"{fmt.format(r[k]):>15s}" for k, fmt in cols))
    print(f"(command vx = {vx}; h(x) min over run: n/a -- gait battery)")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vx", type=float, default=VX)
    p.add_argument("--duration", type=float, default=DURATION)
    p.add_argument("--gif", action="store_true")
    a = p.parse_args()
    main(a.vx, a.duration, a.gif)
