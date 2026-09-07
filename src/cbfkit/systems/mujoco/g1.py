"""Unitree G1 humanoid helpers: model loading, ids/keyframes, standup task, friction DR.

The MJCF description is the Menagerie ``unitree_g1`` model with hydrax's
modifications, redistributed under Unitree's BSD-3-Clause license (see
``models/g1/LICENSE`` and ``models/g1/README.md``); meshes are fetched on first
use by :mod:`cbfkit.systems.mujoco.assets`.

The G1's 29 actuators are **position servos** (``<position kp="500">``), so a
control vector is a joint-position target, not a torque.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
from jax import Array
from mujoco import mjx

from .assets import g1_model_dir


def load_g1(*, sim: bool = False, offline: bool = False) -> mujoco.MjModel:
    """Load the G1 scene.

    ``sim=True`` applies hydrax's simulation-model overrides (stiffer contact
    and a finer step than the planner model): ``timestep=0.01`` and
    ``o_solimp=[0.9, 0.95, 0.001, 0.5, 2]``. hydrax enables the OVERRIDE flag,
    which MJX does not implement, so the same effect is applied per geom: every
    geom's ``solimp/solref/friction/margin/gap`` is replaced by the ``o_*``
    option values -- exactly what OVERRIDE does inside ``mj_step``.
    """
    m = mujoco.MjModel.from_xml_path(str(g1_model_dir(offline=offline) / "scene.xml"))
    if sim:
        m.opt.timestep = 0.01
        m.opt.o_solimp[:] = [0.9, 0.95, 0.001, 0.5, 2]
        m.geom_solimp[:] = m.opt.o_solimp
        m.geom_solref[:] = m.opt.o_solref
        m.geom_friction[:] = m.opt.o_friction[:3]
        m.geom_margin[:] = m.opt.o_margin
        m.geom_gap[:] = 0.0
    return m


class G1:
    """Ids and keyframes for a loaded G1 model (any ``load_g1`` variant)."""

    def __init__(self, mj_model: mujoco.MjModel) -> None:
        self.mj_model = mj_model
        self.torso_site = int(mj_model.site("imu_in_torso").id)
        self.left_foot_site = int(mj_model.site("left_foot").id)
        self.right_foot_site = int(mj_model.site("right_foot").id)
        self.pelvis_body = int(mj_model.body("pelvis").id)
        self.q_stand = jnp.asarray(mj_model.keyframe("stand").qpos)
        self.nq = int(mj_model.nq)
        self.nv = int(mj_model.nv)

    # -- initial states -----------------------------------------------------
    def x_stand(self, plant: Any) -> Array:
        """Flat plant state at the ``stand`` keyframe (zero velocity)."""
        x = jnp.zeros(plant.state_dim).at[: self.nq].set(self.q_stand)
        return plant.to_state(plant.from_state(x))

    def x_fallen(self, plant: Any) -> Array:
        """Stand keyframe rotated onto its side (hydrax's standup initial state)."""
        q = self.q_stand.at[3:7].set(jnp.array([0.7, 0.0, -0.7, 0.0]))
        x = jnp.zeros(plant.state_dim).at[: self.nq].set(q)
        return plant.to_state(plant.from_state(x))

    # -- observables ---------------------------------------------------------
    def torso_height(self, data: mjx.Data) -> Array:
        return data.site_xpos[self.torso_site, 2]

    def torso_up_vector(self, data: mjx.Data) -> Array:
        """World-frame direction of the torso's z-axis (3,)."""
        return data.site_xmat[self.torso_site].reshape(3, 3)[:, 2]

    def torso_upright(self, data: mjx.Data) -> Array:
        """cos(angle between torso z-axis and world up): 1 upright, 0 horizontal."""
        return self.torso_up_vector(data)[2]


def standup_costs(
    g1: G1,
    target_height: float = 0.9,
    w_orientation: float = 10.0,
    w_height: float = 10.0,
    w_posture: float = 0.1,
) -> Tuple[Callable[[mjx.Data, Array, Any], Array], Callable[[mjx.Data, Any], Array]]:
    """hydrax's ``HumanoidStandup`` costs: orientation + height + nominal posture.

    Orientation is ``||R e_z - e_z||^2`` (0 when upright, 2 when horizontal),
    computed from the torso site frame rather than a sensor so it works on the
    stock Menagerie model too.
    """
    up = jnp.array([0.0, 0.0, 1.0])
    q_stand_joints = g1.q_stand[7:]

    def running(data: mjx.Data, u: Array, aux: Any) -> Array:
        orientation = jnp.sum(jnp.square(g1.torso_up_vector(data) - up))
        height = jnp.square(g1.torso_height(data) - target_height)
        posture = jnp.sum(jnp.square(data.qpos[7:] - q_stand_joints))
        return w_orientation * orientation + w_height * height + w_posture * posture

    def terminal(data: mjx.Data, aux: Any) -> Array:
        return running(data, jnp.zeros(0), aux)

    return running, terminal


def friction_randomizer(
    low: float = 0.5, high: float = 2.0
) -> Callable[[mjx.Model, Array], Dict[str, Array]]:
    """Scale every geom's sliding friction by ``U(low, high)`` (hydrax's G1 randomisation)."""

    def randomize(model: mjx.Model, key: Array) -> Dict[str, Array]:
        n = model.geom_friction.shape[0]
        scale = jax.random.uniform(key, (n,), minval=low, maxval=high)
        return {
            "geom_friction": model.geom_friction.at[:, 0].set(model.geom_friction[:, 0] * scale)
        }

    return randomize


def walk_costs(
    g1: G1,
    *,
    target_height: float = 0.9,
    w_velocity: float = 10.0,
    w_orientation: float = 10.0,
    w_height: float = 5.0,
    w_posture: float = 0.1,
    w_angvel: float = 0.1,
    w_control: float = 0.0,
    w_yaw_rate: float = 1.0,
    w_balance: float = 0.0,
    w_fall: float = 0.0,
    h_min: float = 0.75,
    w_feet: float = 0.0,
    foot_z_max: float = 0.15,
    w_qvel: float = 0.0,
    w_gait: float = 0.0,
    gait_freq: float = 1.5,
    gait_swing_height: float = 0.08,
    gait_duty: float = 0.5,
    foot_z0: float = 0.033,
) -> Tuple[Callable[[mjx.Data, Array, Any], Array], Callable[[mjx.Data, Any], Array]]:
    """Velocity-tracking ("walk") costs in the spirit of MJPC's humanoid walk task.

    ``w_fall`` is a hinge on torso height: ``w_fall * max(0, h_min - h)^2``.
    ``w_feet`` is a hinge on foot-site height above ``foot_z_max`` (stops the
    sampler from "walking" by kicking the legs up).
    ``w_gait`` tracks a periodic foot-height reference (DIAL-MPC-style gait
    prior): each foot is in stance (z = ``foot_z0``) for a fraction ``gait_duty``
    of the period ``1/gait_freq`` and follows a half-sine swing of height
    ``gait_swing_height`` otherwise; the right foot is half a period behind the
    left. Uses ``data.time`` -- the MPC stamps absolute time on the rollout root.
    Unlike the quadratic ``w_height`` term it is indifferent above ``h_min`` and
    steep below it, so a lunge that ends on the knees is never "worth it".

    ``w_balance`` penalises the planar distance between the whole-body CoM and
    the midpoint of the two foot sites (MJPC's "balance" residual).

    ``aux`` is the planar velocity command ``(vx, vy)`` in the world frame (the
    2-D reduced-order control the CBF filters). Tracking is on the pelvis
    free-joint velocity ``qvel[0:2]``; the remaining terms keep the robot
    upright at height, near its standing posture, and not spinning. No gait is
    prescribed -- the sampler has to find one.
    """
    up = jnp.array([0.0, 0.0, 1.0])
    q_stand_joints = g1.q_stand[7:]

    def running(data: mjx.Data, u: Array, aux: Any) -> Array:
        v_cmd = jnp.zeros(2) if aux is None else jnp.asarray(aux)[:2]
        velocity = jnp.sum(jnp.square(data.qvel[0:2] - v_cmd))
        orientation = jnp.sum(jnp.square(g1.torso_up_vector(data) - up))
        height = jnp.square(g1.torso_height(data) - target_height)
        posture = jnp.sum(jnp.square(data.qpos[7:] - q_stand_joints))
        angvel = jnp.sum(jnp.square(data.qvel[3:5]))  # roll/pitch rates
        yaw_rate = jnp.square(data.qvel[5])
        feet_mid = 0.5 * (data.site_xpos[g1.left_foot_site] + data.site_xpos[g1.right_foot_site])
        balance = jnp.sum(jnp.square(data.subtree_com[0, :2] - feet_mid[:2]))
        fall = jnp.square(jnp.maximum(0.0, h_min - g1.torso_height(data)))
        feet_z = jnp.array(
            [data.site_xpos[g1.left_foot_site, 2], data.site_xpos[g1.right_foot_site, 2]]
        )
        feet = jnp.sum(jnp.square(jnp.maximum(0.0, feet_z - foot_z_max)))
        joint_vel = jnp.sum(jnp.square(data.qvel[6:]))  # energy-like smoothing of the flailing
        gait: Array = jnp.zeros(())
        if w_gait > 0.0:

            def z_ref(offset):
                phase = jnp.mod(gait_freq * data.time + offset, 1.0)
                swing = jnp.maximum(0.0, phase - gait_duty) / (1.0 - gait_duty)  # 0 in stance
                return foot_z0 + gait_swing_height * jnp.sin(jnp.pi * swing) * (phase > gait_duty)

            gait = jnp.square(feet_z[0] - z_ref(0.0)) + jnp.square(feet_z[1] - z_ref(0.5))
        control = (
            jnp.sum(jnp.square(u - data.qpos[7:])) if u.shape[0] == q_stand_joints.shape[0] else 0.0
        )
        return (
            w_velocity * velocity
            + w_orientation * orientation
            + w_height * height
            + w_posture * posture
            + w_angvel * angvel
            + w_yaw_rate * yaw_rate
            + w_balance * balance
            + w_fall * fall
            + w_feet * feet
            + w_qvel * joint_vel
            + w_gait * gait
            + w_control * control
        )

    def terminal(data: mjx.Data, aux: Any) -> Array:
        return running(data, jnp.zeros(0), aux)

    return running, terminal


__all__ = ["G1", "friction_randomizer", "load_g1", "standup_costs", "walk_costs"]
