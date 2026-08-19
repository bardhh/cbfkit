"""Unitree's pretrained G1 walking policy (``unitree_rl_gym``) as a CBFKit locomotion controller.

Source: https://github.com/unitreerobotics/unitree_rl_gym (BSD-3-Clause, Unitree
Robotics) -- ``deploy/pre_train/g1/motion.pt`` with ``deploy/deploy_mujoco/
{deploy_mujoco.py, configs/g1.yaml}`` and the 12-DoF legs-only MJCF
``resources/robots/g1_description/g1_12dof.xml`` (waist and arms fixed). All
files are fetched on first use from a pinned commit and SHA-256 verified
(:func:`cbfkit.systems.mujoco.assets.unitree_rl_gym_dir`).

The checkpoint is a TorchScript ``PolicyExporterLSTM``: obs(47) -> LSTM(64) ->
Linear(64,32) -> ELU -> Linear(32,12). It is read here without ``torch`` (the
archive is a zip of a pickle plus raw float32 blobs) and evaluated in JAX, so
it runs inside the simulator's ``lax.scan`` with its LSTM state carried in
``ControllerData.sub_data["_policy"]``.

Observation (from ``deploy_mujoco.py``), 47 entries::

    [ omega_body * 0.25 (3) | gravity_in_body (3) | cmd * (2, 2, 0.25) (3) |
      (q - q_default) * 1.0 (12) | dq * 0.05 (12) | last action (12) |
      sin, cos of a 0.8 s phase clock (2) ]

Action (12) -> joint targets ``q_target = 0.25 * action + q_default``; the plant
applies ``tau = kp (q_target - q) - kd dq`` at every 2 ms substep (50 Hz policy,
500 Hz PD), i.e. :func:`make_g1_12dof_plant`.

Trained in Isaac Gym, deployed in MuJoCo by Unitree; MJX here is sim2sim2.
"""

import io
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

from .assets import unitree_rl_gym_dir
from .plant import MujocoPlant

# --------------------------------------------------------------------------- config
# From deploy/deploy_mujoco/configs/g1.yaml at the pinned commit (kept here so
# the controller does not depend on PyYAML).
G1_12DOF_CONFIG: Dict[str, Any] = {
    "simulation_dt": 0.002,
    "control_decimation": 10,
    "kps": [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
    "kds": [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
    "default_angles": [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    "ang_vel_scale": 0.25,
    "dof_pos_scale": 1.0,
    "dof_vel_scale": 0.05,
    "action_scale": 0.25,
    "cmd_scale": [2.0, 2.0, 0.25],
    "num_actions": 12,
    "num_obs": 47,
    "phase_period": 0.8,
}
_REL_XML = "resources/robots/g1_description/scene.xml"
_REL_POLICY = "deploy/pre_train/g1/motion.pt"


# --------------------------------------------------------------------------- torch-free loader
class _Storage:
    def __init__(self, key: str, dtype: Any) -> None:
        self.key, self.dtype = key, dtype


class _Opaque:
    """Stand-in for torch classes we do not need (modules, typed lists, ...)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args

    def __setstate__(self, state: Any) -> None:
        self.state = state


def load_torchscript_tensors(path: Path) -> Dict[str, np.ndarray]:
    """Read every tensor of a TorchScript archive into ``{"dotted.name": ndarray}`` without torch."""
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0] + "/"

    def rebuild(storage: _Storage, offset: int, size: Any, stride: Any, *_: Any) -> np.ndarray:
        buf = np.frombuffer(z.read(prefix + "data/" + storage.key), dtype=storage.dtype)
        strides = tuple(int(s) * buf.itemsize for s in stride)
        return np.lib.stride_tricks.as_strided(
            buf[offset:], shape=tuple(size), strides=strides
        ).copy()

    dtypes = {
        "FloatStorage": np.float32,
        "DoubleStorage": np.float64,
        "HalfStorage": np.float16,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
    }

    class Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name in dtypes:
                return name
            if (module, name) == ("collections", "OrderedDict"):
                import collections

                return collections.OrderedDict
            return _Opaque

        def persistent_load(self, pid: Any) -> _Storage:
            _, storage_type, key, _location, _numel = pid
            name = storage_type if isinstance(storage_type, str) else storage_type.__name__
            return _Storage(key, dtypes[name])

    root = Unpickler(io.BytesIO(z.read(prefix + "data.pkl"))).load()
    tensors: Dict[str, np.ndarray] = {}

    def walk(o: Any, name: str) -> None:
        if isinstance(o, np.ndarray):
            tensors[name] = o
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{name}.{k}" if name else str(k))
        elif hasattr(o, "state"):
            walk(o.state, name)

    walk(root, "")
    return tensors


class PolicyParams(NamedTuple):
    w_ih: Array  # (4H, 47)
    w_hh: Array  # (4H, H)
    b_ih: Array  # (4H,)
    b_hh: Array  # (4H,)
    w1: Array  # (32, H)
    b1: Array  # (32,)
    w2: Array  # (12, 32)
    b2: Array  # (12,)


def load_g1_policy_params(path: Optional[Path] = None, offline: bool = False) -> PolicyParams:
    """Load ``motion.pt`` (fetched if needed) into JAX arrays."""
    if path is None:
        path = unitree_rl_gym_dir(offline=offline) / _REL_POLICY
    t = load_torchscript_tensors(Path(path))
    f = lambda k: jnp.asarray(t[k], dtype=jnp.float32)  # noqa: E731
    return PolicyParams(
        w_ih=f("memory.weight_ih_l0"),
        w_hh=f("memory.weight_hh_l0"),
        b_ih=f("memory.bias_ih_l0"),
        b_hh=f("memory.bias_hh_l0"),
        w1=f("actor.0.weight"),
        b1=f("actor.0.bias"),
        w2=f("actor.2.weight"),
        b2=f("actor.2.bias"),
    )


# --------------------------------------------------------------------------- plant
def load_g1_12dof(offline: bool = False) -> mujoco.MjModel:
    """The 12-DoF legs-only G1 (upper body fixed) with the deploy timestep (2 ms)."""
    m = mujoco.MjModel.from_xml_path(str(unitree_rl_gym_dir(offline=offline) / _REL_XML))
    if (m.nq, m.nv, m.nu) != (19, 18, 12):
        raise RuntimeError(
            f"unexpected g1_12dof model dims {(m.nq, m.nv, m.nu)}; observation layout assumes (19, 18, 12)"
        )
    m.opt.timestep = G1_12DOF_CONFIG["simulation_dt"]
    # MJX collision support: it has no cylinder-mesh pair, and full mesh-mesh
    # self-collision among 27 convex hulls is needlessly heavy. Keep what walking
    # depends on -- the four small foot spheres against the floor -- and let body
    # meshes collide with the floor only (contype=2, conaffinity=1: pairs only
    # with the floor's contype=1/conaffinity=1, never with each other).
    # Head/arm/hand meshes are dropped from the collision set entirely (they only
    # matter after a fall and are the ones MJX warns about).
    for gid in range(m.ngeom):
        gtype = m.geom_type[gid]
        if m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0:
            continue  # visual-only
        if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            m.geom_contype[gid] = 0
            m.geom_conaffinity[gid] = 0
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gid]) or ""
            keep = any(k in mesh_name for k in ("hip", "knee", "ankle", "pelvis", "torso"))
            m.geom_contype[gid] = 2 if keep else 0
            m.geom_conaffinity[gid] = 1 if keep else 0
    return m


def make_g1_12dof_plant(offline: bool = False) -> MujocoPlant:
    """``MujocoPlant`` for the 12-DoF G1: ``u`` = 12 joint targets, PD torque at every 2 ms substep."""
    m = load_g1_12dof(offline=offline)
    kp = jnp.asarray(G1_12DOF_CONFIG["kps"], dtype=float)
    kd = jnp.asarray(G1_12DOF_CONFIG["kds"], dtype=float)

    def pd(data, target_q):
        return kp * (target_q - data.qpos[7:19]) - kd * data.qvel[6:18]

    return MujocoPlant(m, substeps=G1_12DOF_CONFIG["control_decimation"], ctrl_map=pd, nu=12)


# --------------------------------------------------------------------------- policy
class PolicyState(NamedTuple):
    h: Array  # (H,)
    c: Array  # (H,)
    last_action: Array  # (12,)


def _gravity_in_body(quat: Array) -> Array:
    """``deploy_mujoco.get_gravity_orientation``: world -z expressed in the body frame."""
    qw, qx, qy, qz = quat
    return jnp.array(
        [2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)]
    )


class UnitreeG1WalkPolicy:
    """Unitree's LSTM walking policy as a CBFKit ``ControllerCallable`` factory."""

    def __init__(self, params: Optional[PolicyParams] = None, offline: bool = False) -> None:
        self.params = params if params is not None else load_g1_policy_params(offline=offline)
        c = G1_12DOF_CONFIG
        self.q_default = jnp.asarray(c["default_angles"], dtype=float)
        self.ang_vel_scale = float(c["ang_vel_scale"])
        self.dof_pos_scale = float(c["dof_pos_scale"])
        self.dof_vel_scale = float(c["dof_vel_scale"])
        self.action_scale = float(c["action_scale"])
        self.cmd_scale = jnp.asarray(c["cmd_scale"], dtype=float)
        self.period = float(c["phase_period"])
        self.hidden = int(self.params.w_hh.shape[1])
        self.n_act = int(self.params.w2.shape[0])

    def init_state(self) -> PolicyState:
        return PolicyState(
            h=jnp.zeros(self.hidden), c=jnp.zeros(self.hidden), last_action=jnp.zeros(self.n_act)
        )

    def observation(self, x: Array, cmd: Array, t: Array, last_action: Array) -> Array:
        """47-dim observation from the flat plant state ``[qpos(19) | qvel(18) | com]``."""
        quat = x[3:7]
        omega = x[19 + 3 : 19 + 6]
        qj = x[7:19]
        dqj = x[19 + 6 : 19 + 18]
        phase = jnp.mod(t, self.period) / self.period
        return jnp.concatenate(
            [
                omega * self.ang_vel_scale,
                _gravity_in_body(quat),
                jnp.asarray(cmd, dtype=float)[:3] * self.cmd_scale,
                (qj - self.q_default) * self.dof_pos_scale,
                dqj * self.dof_vel_scale,
                last_action,
                jnp.array([jnp.sin(2 * jnp.pi * phase), jnp.cos(2 * jnp.pi * phase)]),
            ]
        ).astype(jnp.float32)

    def forward(self, obs: Array, state: PolicyState) -> Tuple[Array, PolicyState]:
        p = self.params
        gates = p.w_ih @ obs + p.b_ih + p.w_hh @ state.h + p.b_hh
        i, f, g, o = jnp.split(gates, 4)
        c = jax.nn.sigmoid(f) * state.c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        z = jax.nn.elu(p.w1 @ h + p.b1)
        action = p.w2 @ z + p.b2
        return action, PolicyState(h=h, c=c, last_action=action)

    def step(self, x: Array, cmd: Array, t: Array, state: PolicyState) -> Tuple[Array, PolicyState]:
        """One 50 Hz policy step: returns the 12 joint-position targets and the new state."""
        obs = self.observation(x, cmd, t, state.last_action)
        action, state = self.forward(obs, state)
        return self.action_scale * action + self.q_default, state

    def as_controller(self):
        """``(t, x, u_nom, key, data) -> (q_target, data)``; ``u_nom[:2]`` is the (vx, vy) command.

        A third command entry (yaw rate) is used when ``u_nom`` has 3 entries,
        otherwise 0. State lives in ``sub_data["_policy"]`` (carry-only).
        """
        if getattr(self, "_controller", None) is not None:
            return self._controller

        def controller(t, x, u_nom, key, data):
            sub = dict(data.sub_data) if data.sub_data is not None else {}
            state = sub.get("_policy")
            if state is None:
                state = self.init_state()
            u_nom = jnp.asarray(u_nom, dtype=float)
            cmd = jnp.zeros(3).at[: min(3, u_nom.shape[0])].set(u_nom[:3])
            u, state = self.step(x, cmd, t, state)
            sub["_policy"] = state
            return u, data._replace(sub_data=sub, u=u, u_nom=u_nom)

        controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
        self._controller = controller
        return controller


def x0_standing(plant: MujocoPlant) -> Array:
    """Deploy's initial state: model ``qpos0`` (pelvis at its XML height, joints at zero)."""
    return plant.to_state(plant.make_data())


__all__ = [
    "G1_12DOF_CONFIG",
    "PolicyParams",
    "PolicyState",
    "UnitreeG1WalkPolicy",
    "load_g1_12dof",
    "load_g1_policy_params",
    "load_torchscript_tensors",
    "make_g1_12dof_plant",
    "x0_standing",
]
