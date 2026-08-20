"""NVIDIA GR00T GEAR-WBC G1 policy (decoupled whole-body control) -- torch/onnx-free JAX.

The released ``GR00T-WholeBodyControl`` "decoupled WBC" checkpoints
(`NVlabs/GR00T-WholeBodyControl <https://github.com/NVlabs/GR00T-WholeBodyControl>`_,
Apache-2.0 code + NVIDIA Open Model License for the weights, pinned via
``assets.groot_dir``): two stateless ONNX policies -- **Balance** (used while the
locomotion command is ~zero) and **Walk** -- over the 29-DoF G1 (12 legs + 3 waist +
14 arms), producing 15 joint targets (legs + waist) at 50 Hz; arms are PD-held at zero
by the runner. The command carries posture too::

    [vx, vy, wz | height (absolute, default 0.74) | torso roll, pitch, yaw]

Weights are read **without onnx/onnxruntime/torch**: a minimal protobuf wire-format
reader (:func:`load_onnx_tensors`) pulls the initializers out of the ONNX file, and the
network -- recovered from the ONNX graph and pinned by a parity test against
onnxruntime (dev-only oracle) -- is evaluated in JAX::

    estimator: 516 -> 256 -> 256 -> 35 (ELU)          # obs history -> [v_hat(3) | z(32)]
    actor: [obs_last(86) | v_hat | z/||z||] = 121 -> 512 -> 256 -> 256 -> 15 (ELU)

Observation (86, from ``sim2mujoco/run_mujoco_gear_wbc.py``):
``[cmd(7) = loco*[2,2,0.5], height, rpy | omega*0.5 | gravity-in-body |
q - q_default (29) | dq*0.05 (29) | last_action(15)]``; a 6-frame history (current frame
included) is the 516-dim policy input. Deployment constants from ``g1_gear_wbc.yaml``:
sim dt 5 ms x 4 substeps, PD kp ``[150,150,150,200,40,40]x2 + [250]x3``, kd
``[2,2,2,4,2,2]x2 + [5]x3``; arm PD kp 100 / kd 0.5 to the zero pose.

Repo drift note: the upstream sim2mujoco XML carries 43 actuated joints (hands), which
overflows the scripts' hard-coded 86-dim observation -- the working layout needs exactly
29 joints, so the plant here uses the same repo's ``model_data/g1/g1_29dof_old.xml``
(joint order verified: legs, waist, arms -- matching ``default_angles``).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

from cbfkit.systems.mujoco.assets import groot_dir
from cbfkit.systems.mujoco.plant import MujocoPlant
from cbfkit.systems.mujoco.unitree_policy import _gravity_in_body, _wrap, _yaw

# --------------------------------------------------------------------------- deployment constants
GROOT_CONFIG: Dict[str, Any] = {
    "simulation_dt": 0.005,
    "control_decimation": 4,
    "kps": [150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250],
    "kds": [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5],
    "arm_kp": 100.0,
    "arm_kd": 0.5,
    "default_angles": [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
    + [0.0, 0.0, 0.0],
    "ang_vel_scale": 0.5,
    "dof_vel_scale": 0.05,
    "action_scale": 0.25,
    "cmd_scale": [2.0, 2.0, 0.5],
    "height_cmd": 0.74,
    "num_actions": 15,
    "num_dofs": 29,
    "n_obs": 86,
    "obs_history_len": 6,
    "balance_threshold": 0.05,  # |loco_cmd| below this -> Balance policy
}

_REL_XML = "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
_REL_POLICY = "decoupled_wbc/sim2mujoco/resources/robots/g1/policy"


# --------------------------------------------------------------------------- ONNX reading
def _pb_fields(b: bytes):
    """Yield ``(field_no, wire_type, value)`` over a protobuf buffer (wire format only)."""
    i = 0
    n = len(b)
    while i < n:
        tag = 0
        s = 0
        while True:
            v = b[i]
            i += 1
            tag |= (v & 0x7F) << s
            if not v & 0x80:
                break
            s += 7
        f, w = tag >> 3, tag & 7
        if w == 0:
            x = 0
            s = 0
            while True:
                v = b[i]
                i += 1
                x |= (v & 0x7F) << s
                if not v & 0x80:
                    break
                s += 7
            yield f, w, x
        elif w == 2:
            ln = 0
            s = 0
            while True:
                v = b[i]
                i += 1
                ln |= (v & 0x7F) << s
                if not v & 0x80:
                    break
                s += 7
            yield f, w, b[i : i + ln]
            i += ln
        elif w == 5:
            yield f, w, b[i : i + 4]
            i += 4
        elif w == 1:
            yield f, w, b[i : i + 8]
            i += 8
        else:  # pragma: no cover - not produced by ONNX exporters
            raise ValueError(f"unsupported wire type {w}")


def load_onnx_tensors(path: Path) -> Dict[str, np.ndarray]:
    """Read every initializer of an ONNX file into ``{name: ndarray}`` without onnx/torch.

    Parses only the protobuf wire format: ModelProto.graph(7) -> GraphProto.initializer(5)
    -> TensorProto {dims=1, data_type=2, float_data=4, name=8, raw_data=9}.
    """
    buf = Path(path).read_bytes()
    graph = None
    for f, _, v in _pb_fields(buf):
        if f == 7:
            graph = v
    if graph is None:
        raise ValueError(f"{path}: no graph in ONNX model")
    dtypes = {1: np.float32, 6: np.int32, 7: np.int64, 11: np.float64}
    out: Dict[str, np.ndarray] = {}
    for f, _, v in _pb_fields(graph):
        if f != 5:  # initializer
            continue
        dims, dtype, name, raw, fdata = [], 1, None, None, []
        for f2, w2, v2 in _pb_fields(v):
            if f2 == 1:
                dims.append(v2)
            elif f2 == 2:
                dtype = v2
            elif f2 == 8:
                name = v2.decode()
            elif f2 == 9:
                raw = v2
            elif f2 == 4:
                fdata.append(struct.unpack("<f", v2)[0])
        if raw is not None:
            arr = np.frombuffer(raw, dtype=dtypes[dtype])
        else:
            arr = np.asarray(fdata, dtype=np.float32)
        out[name] = arr.reshape(dims) if dims else arr
    return out


class GrootParams(NamedTuple):
    balance: Dict[str, Array]
    walk: Dict[str, Array]


def load_groot_params(offline: bool = False) -> GrootParams:
    d = groot_dir(offline=offline)
    expect = {
        "estimator.0.weight": (256, 516),
        "estimator.2.weight": (256, 256),
        "estimator.4.weight": (35, 256),
        "actor.0.weight": (512, 121),
        "actor.2.weight": (256, 512),
        "actor.4.weight": (256, 256),
        "actor.6.weight": (15, 256),
    }

    def load(name: str) -> Dict[str, Array]:
        t = load_onnx_tensors(d / _REL_POLICY / name)
        for k, shape in expect.items():
            if t[k].shape != shape:
                raise RuntimeError(f"{name}: {k} has shape {t[k].shape}, expected {shape}")
        return {k: jnp.asarray(v) for k, v in t.items()}

    return GrootParams(
        balance=load("GR00T-WholeBodyControl-Balance.onnx"),
        walk=load("GR00T-WholeBodyControl-Walk.onnx"),
    )


# --------------------------------------------------------------------------- forward
def policy_forward(p: Dict[str, Array], obs_hist: Array) -> Array:
    """One GEAR-WBC policy: ``(6, 86)`` history (oldest first, current frame last) -> 15 actions."""
    x = jnp.asarray(obs_hist, dtype=jnp.float32).reshape(-1)  # (516,)
    e = jax.nn.elu(x @ p["estimator.0.weight"].T + p["estimator.0.bias"])
    e = jax.nn.elu(e @ p["estimator.2.weight"].T + p["estimator.2.bias"])
    e = e @ p["estimator.4.weight"].T + p["estimator.4.bias"]  # (35,)
    v_hat, z = e[:3], e[3:]
    z = z / jnp.maximum(jnp.linalg.norm(z), 1e-12)
    a = jnp.concatenate([obs_hist[-1].astype(jnp.float32), v_hat, z])  # (121,)
    a = jax.nn.elu(a @ p["actor.0.weight"].T + p["actor.0.bias"])
    a = jax.nn.elu(a @ p["actor.2.weight"].T + p["actor.2.bias"])
    a = jax.nn.elu(a @ p["actor.4.weight"].T + p["actor.4.bias"])
    return a @ p["actor.6.weight"].T + p["actor.6.bias"]


# --------------------------------------------------------------------------- plant
def load_g1_29dof(offline: bool = False) -> mujoco.MjModel:
    """The repo's 29-DoF G1 (``g1_29dof_old.xml``) at the GEAR-WBC timestep (5 ms).

    The upstream file is a robot-only model (no floor); a ground plane is injected into a
    patched copy next to it (so the relative ``meshdir`` still resolves) before loading.
    """
    src = groot_dir(offline=offline) / _REL_XML
    patched = src.with_name("g1_29dof_floor.xml")
    text = src.read_text()
    if 'type="plane"' not in text:
        text = text.replace(
            "<worldbody>",
            '<worldbody>\n    <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" condim="3"/>'
            '\n    <light pos="0 0 3" dir="0 0 -1" directional="true"/>',
            1,
        )
    if not patched.exists() or patched.read_text() != text:
        patched.write_text(text)
    m = mujoco.MjModel.from_xml_path(str(patched))
    if (m.nq, m.nv, m.nu) != (36, 35, 29):
        raise RuntimeError(
            f"unexpected g1_29dof model dims {(m.nq, m.nv, m.nu)}; observation layout assumes (36, 35, 29)"
        )
    m.opt.timestep = GROOT_CONFIG["simulation_dt"]
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    # MJX collision trim, as for the other G1 plants: keep foot geoms + floor.
    for gid in range(m.ngeom):
        if m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0:
            continue
        gtype = m.geom_type[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gid]) or ""
            keep = any(k in mesh_name for k in ("ankle_roll", "pelvis"))
            # contype=2/conaffinity=1: collide with the floor (contype 1) only, never
            # with each other -- as in load_g1_12dof.
            m.geom_contype[gid] = 2 if keep else 0
            m.geom_conaffinity[gid] = 1 if keep else 0
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            m.geom_contype[gid] = 0
            m.geom_conaffinity[gid] = 0
    return m


def make_g1_29dof_plant(offline: bool = False) -> MujocoPlant:
    """``MujocoPlant`` for GEAR-WBC: ``u`` = 15 leg/waist targets; arms PD-held at zero."""
    m = load_g1_29dof(offline=offline)
    c = GROOT_CONFIG
    kp = jnp.asarray(c["kps"], dtype=float)
    kd = jnp.asarray(c["kds"], dtype=float)
    akp, akd = float(c["arm_kp"]), float(c["arm_kd"])

    def pd(data, target_q):
        q, dq = data.qpos[7:36], data.qvel[6:35]
        leg_tau = kp * (target_q - q[:15]) - kd * dq[:15]
        arm_tau = akp * (0.0 - q[15:]) - akd * dq[15:]
        return jnp.concatenate([leg_tau, arm_tau])

    return MujocoPlant(m, substeps=c["control_decimation"], ctrl_map=pd, nu=15)


def x0_standing(plant: MujocoPlant) -> Array:
    """Standing start: pelvis at the command height, legs at the default angles."""
    c = GROOT_CONFIG
    qpos = jnp.zeros(plant.nq).at[2].set(0.76).at[3].set(1.0)
    qpos = qpos.at[7:22].set(jnp.asarray(c["default_angles"], dtype=float))
    return plant.to_state(plant.from_state(jnp.concatenate([qpos, jnp.zeros(plant.nv + 3)])))


# --------------------------------------------------------------------------- controller
class GrootState(NamedTuple):
    """Carry of one GEAR-WBC control step (rides in ``sub_data["_groot"]``)."""

    hist: Array  # (6, 86) observation history, oldest first, current frame last
    last_action: Array  # (15,)


TorsoCommand = Union[None, Tuple[float, float, float], Callable[..., Array]]


class GrootGearWbcPolicy:
    """GEAR-WBC (Balance + Walk) as a CBFKit controller factory. See the module docstring."""

    def __init__(self, params: Optional[GrootParams] = None, offline: bool = False) -> None:
        self.params = params if params is not None else load_groot_params(offline=offline)
        c = GROOT_CONFIG
        self.q_default = jnp.zeros(29).at[:15].set(jnp.asarray(c["default_angles"], dtype=float))
        self.cmd_scale = jnp.asarray(c["cmd_scale"], dtype=float)

    def init_state(self) -> GrootState:
        c = GROOT_CONFIG
        return GrootState(
            hist=jnp.zeros((c["obs_history_len"], c["n_obs"])), last_action=jnp.zeros(15)
        )

    def step(self, x: Array, cmd: Array, state: GrootState) -> Tuple[Array, GrootState]:
        """One 50 Hz step. ``x`` = flat plant state ``[qpos(36) | qvel(35) | com]``;
        ``cmd`` = ``[vx, vy, wz, height, roll, pitch, yaw]`` (unscaled; scaling applied
        here as in the deploy script). Returns 15 PD targets + new carry."""
        c = GROOT_CONFIG
        cmd = jnp.asarray(cmd, dtype=float)
        qj = x[7:36]
        dqj = x[36 + 6 : 36 + 35]
        quat = x[3:7]
        omega = x[36 + 3 : 36 + 6]
        obs = jnp.concatenate(
            [
                cmd[:3] * self.cmd_scale,
                cmd[3:7],
                omega * c["ang_vel_scale"],
                _gravity_in_body(quat),
                qj - self.q_default,
                dqj * c["dof_vel_scale"],
                state.last_action,
            ]
        ).astype(jnp.float32)
        hist = jnp.roll(state.hist, -1, axis=0).at[-1].set(obs)
        walking = jnp.linalg.norm(cmd[:3]) > c["balance_threshold"]
        act_w = policy_forward(self.params.walk, hist)
        act_b = policy_forward(self.params.balance, hist)
        action = jnp.where(walking, act_w, act_b)
        target = action * c["action_scale"] + self.q_default[:15]
        return target, GrootState(hist=hist, last_action=action)

    def as_controller(
        self,
        *,
        torso_command: TorsoCommand = None,
        height: float = GROOT_CONFIG["height_cmd"],
        world_frame: bool = True,
        heading_gain: float = 2.0,
        max_yaw_rate: float = 1.0,
        min_speed_for_heading: float = 0.05,
    ):
        """``(t, x, u_nom, key, data) -> (q_target(15), data)``.

        ``u_nom[:2]`` is a planar velocity command (world frame by default), turned into
        body-frame ``(vx, vy)`` plus a heading-following yaw *rate* (``wz``; GEAR-WBC
        takes rates, unlike AMO's absolute target yaw) -- a 3-entry ``u_nom`` supplies
        ``wz`` directly. ``height`` is the absolute pelvis-height command;
        ``torso_command`` adds ``(roll, pitch, yaw)`` -- constant, ``t -> (3,)``, or
        ``(t, x, sub) -> (3,)``. Carry in ``sub_data["_groot"]``; the assembled
        7-command is logged as ``groot_cmd``.
        """
        import inspect

        if torso_command is None:
            torso_fn = lambda t, x, sub: jnp.zeros(3)  # noqa: E731
        elif callable(torso_command):
            n_args = len(inspect.signature(torso_command).parameters)
            torso_fn = (
                torso_command if n_args >= 3 else (lambda t, x, sub: torso_command(t))  # noqa: E731
            )
        else:
            const = jnp.asarray(torso_command, dtype=float)
            torso_fn = lambda t, x, sub: const  # noqa: E731

        def controller(t, x, u_nom, key, data):
            sub = dict(data.sub_data) if data.sub_data is not None else {}
            state = sub.get("_groot")
            if state is None:
                state = self.init_state()
            u_nom = jnp.asarray(u_nom, dtype=float)
            v = u_nom[:2]
            yaw = _yaw(x[3:7])
            if world_frame:
                cs, sn = jnp.cos(yaw), jnp.sin(yaw)
                v_body = jnp.array([cs * v[0] + sn * v[1], -sn * v[0] + cs * v[1]])
            else:
                v_body = v
            if u_nom.shape[0] >= 3:
                wz = u_nom[2]
            else:
                speed = jnp.linalg.norm(v)
                heading = jnp.arctan2(v[1], v[0])
                wz = jnp.clip(heading_gain * _wrap(heading - yaw), -max_yaw_rate, max_yaw_rate)
                wz = jnp.where(speed > min_speed_for_heading, wz, 0.0)
            torso = jnp.asarray(torso_fn(t, x, sub), dtype=float)
            cmd = jnp.concatenate([v_body, jnp.array([wz, height]), torso])
            u, state = self.step(x, cmd, state)
            sub["_groot"] = state
            sub["groot_cmd"] = cmd
            return u, data._replace(sub_data=sub, u=u, u_nom=u_nom)

        controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
        return controller


__all__ = [
    "GROOT_CONFIG",
    "GrootGearWbcPolicy",
    "GrootParams",
    "GrootState",
    "load_g1_29dof",
    "load_groot_params",
    "load_onnx_tensors",
    "make_g1_29dof_plant",
    "policy_forward",
    "x0_standing",
]
