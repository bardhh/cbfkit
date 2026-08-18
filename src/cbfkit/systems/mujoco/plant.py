"""``MujocoPlant``: an MJCF model as a CBFKit plant.

The plant is stateless: it owns the ``mjx.Model`` and exposes pure functions
over ``mjx.Data``. The simulator carries ``mjx.Data`` between steps and logs
``to_state(data)``, a flat vector ``[qpos | qvel | com_xyz]``. The CoM is
appended so reduced-order certificates can index it directly
(``plant.com_indices``) -- the raw ``qpos[0:2]`` of a floating-base robot is
the free-joint position, not the centre of mass.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
from jax import Array
from mujoco import mjx


class MujocoPlant:
    """Wrap a ``mujoco.MjModel`` for use with ``simulator.execute(plant=...)``."""

    def __init__(self, mj_model: mujoco.MjModel, *, substeps: int = 1, com_body: int = 0) -> None:
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model)
        self.substeps = int(substeps)
        self.nq = int(mj_model.nq)
        self.nv = int(mj_model.nv)
        self.nu = int(mj_model.nu)
        self.state_dim = self.nq + self.nv + 3
        self.com_indices: Tuple[int, int] = (self.nq + self.nv, self.nq + self.nv + 1)
        self.dt = float(mj_model.opt.timestep) * self.substeps
        self._com_body = int(com_body)
        limited = jnp.asarray(mj_model.actuator_ctrllimited, dtype=bool)
        rng = jnp.asarray(mj_model.actuator_ctrlrange)
        self.u_min = jnp.where(limited, rng[:, 0], -jnp.inf)
        self.u_max = jnp.where(limited, rng[:, 1], jnp.inf)
        # Jit the hot methods once: called from Python (eager path, tests) they
        # would otherwise re-trace mjx.step on every call; inside an outer
        # jit/scan a jitted callee is inlined at no cost.
        self.step = jax.jit(self.step)  # type: ignore[method-assign]
        self.to_state = jax.jit(self.to_state)  # type: ignore[method-assign]
        self.from_state = jax.jit(self.from_state)  # type: ignore[method-assign]

    # -- data construction -------------------------------------------------
    def make_data(self) -> mjx.Data:
        """Fresh ``mjx.Data`` at the model's default configuration, forward-evaluated."""
        return mjx.forward(self.model, mjx.make_data(self.mj_model))

    def from_state(self, x: Array) -> mjx.Data:
        """Rebuild ``mjx.Data`` from a flat state. Lossy: contact/warm-start state is fresh.

        Only ``qpos`` and ``qvel`` are read; any trailing entries (e.g. the CoM
        appended by ``to_state``) are ignored.
        """
        d = mjx.make_data(self.mj_model)
        d = d.replace(qpos=x[: self.nq], qvel=x[self.nq : self.nq + self.nv])
        return mjx.forward(self.model, d)

    # -- dynamics ----------------------------------------------------------
    def step(self, data: mjx.Data, u: Array, model: Optional[mjx.Model] = None) -> mjx.Data:
        """Advance ``data`` by ``substeps`` MJX steps holding ``ctrl = u``.

        ``model`` overrides the plant's model (used for domain randomisation).
        """
        m = self.model if model is None else model

        def _one(_, d):
            return mjx.step(m, d.replace(ctrl=u))

        return jax.lax.fori_loop(0, self.substeps, _one, data)

    # -- projections -------------------------------------------------------
    def to_state(self, data: mjx.Data) -> Array:
        """Flat state ``[qpos | qvel | com_xyz]``.

        ``mjx.step`` integrates ``qpos`` *after* computing ``subtree_com``, so
        ``data.subtree_com`` lags one step. Recompute kinematics from the
        current ``qpos`` so the logged CoM matches the logged configuration.
        """
        d = mjx.com_pos(self.model, mjx.kinematics(self.model, data))
        return jnp.concatenate([data.qpos, data.qvel, d.subtree_com[self._com_body]])

    # -- helpers -----------------------------------------------------------
    def body_id(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name))

    def body_pos(self, data: mjx.Data, body_id: int) -> Array:
        return data.xpos[body_id]
