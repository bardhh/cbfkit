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
    and a finer step than the planner model): ``timestep=0.01``,
    ``o_solimp=[0.9, 0.95, 0.001, 0.5, 2]`` with the OVERRIDE flag enabled.
    """
    m = mujoco.MjModel.from_xml_path(str(g1_model_dir(offline=offline) / "scene.xml"))
    if sim:
        m.opt.timestep = 0.01
        m.opt.o_solimp[:] = [0.9, 0.95, 0.001, 0.5, 2]
        m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
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


__all__ = ["G1", "friction_randomizer", "load_g1", "standup_costs"]
