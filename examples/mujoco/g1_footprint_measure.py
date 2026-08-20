"""Measure the G1's *anisotropic footprint* and per-axis velocity tracking under AMO.

Feeds the rotating-ellipse footprint CBF (``reduced_order.com_agent_ellipse_hocbfs``):

1. **Footprint**: run stand / forward walk (vx) / sidestep (vy, heading held) in MJX with
   the AMO whole-body policy; forward kinematics per control step; report the body-frame
   extents ``max |body_xy - com_xy|`` split into the longitudinal (facing) and lateral
   (shoulder) axes -- for the *upper body* (bodies above ``UPPER_Z``) and the full body.
   The barrier uses the upper-body ellipse: legs exceed it longitudinally during stride,
   but pedestrian discs describe whole people too and feet interleave when humans squeeze
   through a gap (documented modelling choice, same spirit as the 0.35 m disc it replaces).
2. **Per-axis tracking**: body-frame residuals ``v_com - v_cmd`` for vx-only and vy-only
   command legs (quantiles). The lateral axis is expected to track worse; the ellipse
   CBF's robust margin should use the per-axis bound, or the geometry gain is illusory.

Measured (MJX, CPU, AMO, 12 s per leg, seed 0) -- see the table this script prints;
the adopted barrier constants live in ``reduced_order.G1_FOOTPRINT``.

    python examples/mujoco/g1_footprint_measure.py
"""

import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import cbfkit.simulation.simulator as sim
from cbfkit.systems.mujoco.amo_policy import (
    AmoWholeBodyPolicy,
    make_g1_23dof_plant,
    x0_standing,
)
from cbfkit.utils.user_types import ControllerData, PlannerData

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))

UPPER_Z = 0.6  # m: bodies above this height count as "upper body"
LEG_DURATION = 2.0 if TEST_MODE else 12.0
SETTLE = 0.5 if TEST_MODE else 3.0  # s discarded at the start of each leg
LEGS = [  # (name, world-frame v command, target yaw)
    ("stand", (0.0, 0.0), 0.0),
    ("walk vx=0.4", (0.4, 0.0), 0.0),
    ("sidestep vy=+0.3", (0.0, 0.3), 0.0),  # heading held at 0: pure lateral gait
    ("sidestep vy=-0.3", (0.0, -0.3), 0.0),
]


def run_leg(plant, policy, v_cmd, target_yaw, duration, seed=0):
    ctrl = policy.as_controller()

    def nominal(t, x, key, ref):
        return jnp.array([v_cmd[0], v_cmd[1], target_yaw]), ControllerData()

    res = sim.execute(
        x0=x0_standing(plant),
        dt=plant.dt,
        num_steps=int(round(duration / plant.dt)),
        plant=plant,
        planner_data=PlannerData.from_constant(jnp.array([1e3, 0.0])),
        nominal_controller=nominal,
        controller=ctrl,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=False,
    )
    return np.asarray(res["states"])


def body_frame_extents(plant, states):
    """Per-step body positions in the pelvis-yaw frame relative to the CoM.

    Returns (lon, lat) arrays over (steps x bodies) for the upper body and full body.
    """
    m = plant.mj_model
    d = mujoco.MjData(m)
    n_body = m.nbody
    lon_u, lat_u, lon_f, lat_f = [], [], [], []
    for k in range(len(states)):
        d.qpos[:] = states[k, : plant.nq]
        mujoco.mj_kinematics(m, d)
        com = states[k, plant.com_indices[0] : plant.com_indices[0] + 2]
        qw, qx, qy, qz = states[k, 3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        c, s = np.cos(yaw), np.sin(yaw)
        rel = d.xpos[1:n_body, :2] - com  # skip the world body
        lon = c * rel[:, 0] + s * rel[:, 1]
        lat = -s * rel[:, 0] + c * rel[:, 1]
        upper = d.xpos[1:n_body, 2] > UPPER_Z
        lon_f.append(np.abs(lon).max())
        lat_f.append(np.abs(lat).max())
        lon_u.append(np.abs(lon[upper]).max() if upper.any() else 0.0)
        lat_u.append(np.abs(lat[upper]).max() if upper.any() else 0.0)
    return (np.asarray(lon_u), np.asarray(lat_u), np.asarray(lon_f), np.asarray(lat_f))


def main(seed=0):
    plant = make_g1_23dof_plant()
    policy = AmoWholeBodyPolicy()
    n_settle = int(round(SETTLE / plant.dt))
    print(
        f"{'leg':<18}{'upper lon':>10}{'upper lat':>10}{'full lon':>9}{'full lat':>9}"
        f"{'|v err| mean':>13}{'p95 lon':>8}{'p95 lat':>8}"
    )
    ext_u = {"lon": 0.0, "lat": 0.0}
    for name, v_cmd, tyaw in LEGS:
        S = run_leg(plant, policy, v_cmd, tyaw, LEG_DURATION, seed)[n_settle:]
        lon_u, lat_u, lon_f, lat_f = body_frame_extents(plant, S)
        com = S[:, plant.com_indices[0] : plant.com_indices[0] + 2]
        v_com = np.gradient(com, plant.dt, axis=0) if len(S) > 1 else np.zeros_like(com)
        qw, qx, qy, qz = S[:, 3], S[:, 4], S[:, 5], S[:, 6]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        c, s = np.cos(yaw), np.sin(yaw)
        err_w = v_com - np.asarray(v_cmd)[None]
        err_lon = c * err_w[:, 0] + s * err_w[:, 1]
        err_lat = -s * err_w[:, 0] + c * err_w[:, 1]
        print(
            f"{name:<18}{lon_u.max():>10.2f}{lat_u.max():>10.2f}{lon_f.max():>9.2f}{lat_f.max():>9.2f}"
            f"{np.linalg.norm(err_w, axis=1).mean():>13.2f}"
            f"{np.percentile(np.abs(err_lon), 95):>8.2f}{np.percentile(np.abs(err_lat), 95):>8.2f}"
        )
        ext_u["lon"] = max(ext_u["lon"], float(lon_u.max()))
        ext_u["lat"] = max(ext_u["lat"], float(lat_u.max()))
    print(
        f"upper-body footprint over all gaits: lon {ext_u['lon']:.2f} m, lat {ext_u['lat']:.2f} m "
        f"(disc currently 0.35 m)"
    )
    print("h(x) min over run: n/a (footprint battery)")  # smoke-test marker
    return ext_u


if __name__ == "__main__":
    main()
