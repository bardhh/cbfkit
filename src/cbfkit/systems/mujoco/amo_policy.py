"""AMO whole-body G1 policy (UCSD, RSS 2025) as a CBFKit ``ControllerCallable`` -- torch-free.

AMO (Li, Cheng, Huang, Yang, Qiu, Wang -- `OpenTeleVision/AMO <https://github.com/OpenTeleVision/AMO>`_,
Apache-2.0) is a whole-body policy for the 23-DoF G1 (12 legs + 3 waist + 8 arms) whose command
is *locomotion plus torso posture*::

    [vx, target_yaw, vy, height_delta, torso_yaw, torso_pitch, torso_roll]

so the robot can lean, turn or duck its torso *while walking* -- in-distribution ranges
vx [-0.5, 0.5], vy [-0.4, 0.4], height 0.75 + [-0.5, 0.8] m, torso yaw [-1.57, 1.57],
pitch [-0.52, 1.57], roll [-0.7, 0.7] (O.O.D. combinations degrade gracefully).

The pinned upstream files (``assets.amo_dir``) are read **without torch**: the two TorchScript
archives via :func:`cbfkit.systems.mujoco.unitree_policy.load_torchscript_tensors` and the
norm-stats ``torch.save`` pickle via a restricted numpy unpickler. Inference is pure JAX and
replicates ``play_amo.py`` (the authors' MuJoCo deployment) exactly:

* 50 Hz policy over a 2 ms / 10-substep PD plant (:func:`make_g1_23dof_plant`; play_amo's
  kp/kd/torque limits, Python-side PD on raw torque actuators);
* observation (93): ``[gyro*0.25, roll, pitch, sin/cos(yaw - target_yaw), q - q_default,
  dq*0.05 (ankle/waist-roll/pitch rows zeroed), last_action(23), sin(2*pi*gait), adapter(15)]``
  where the *adapter* MLP (12 -> 15, BatchNorm eval) turns ``[height, torso ypr, arm q(8)]``
  into a whole-body reference;
* policy = ELU MLPs: per-frame encoder + strided Conv1d over the 10-frame history,
  a text-feature encoder/merger over the *last 4* history frames, and a 2474 -> 1024 ->
  1024 -> 512 -> 15 student backbone fed with a 25-frame extra history (which, unlike the
  10-frame one, includes the current frame -- a play_amo subtlety preserved here);
* action: 15 targets (legs + waist) scaled by 0.25 around the defaults; arms PD-held at the
  default pose; gait clock 1.3 Hz with play_amo's stand/walk phase latching.

Histories, gait phase, the stand flag and the followed heading ride in
``ControllerData.sub_data["_amo"]`` (carry-only), so the controller is a pure function and
JIT/scan-safe. MJX note: the upstream XML requests the PGS solver (unsupported in MJX);
:func:`load_g1_23dof` switches to Newton and keeps everything else.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

from cbfkit.systems.mujoco.assets import amo_dir
from cbfkit.systems.mujoco.plant import MujocoPlant
from cbfkit.systems.mujoco.unitree_policy import _wrap, _yaw, load_torchscript_tensors

# --------------------------------------------------------------------------- play_amo constants
AMO_CONFIG: Dict[str, Any] = {
    "simulation_dt": 0.002,
    "control_decimation": 10,
    "kps": [150, 150, 150, 300, 80, 20, 150, 150, 150, 300, 80, 20]
    + [400, 400, 400]
    + [80, 80, 40, 60, 80, 80, 40, 60],
    "kds": [2, 2, 2, 4, 2, 1, 2, 2, 2, 4, 2, 1] + [15, 15, 15] + [2, 2, 1, 1, 2, 2, 1, 1],
    "torque_limits": [88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50]
    + [88, 50, 50]
    + [25, 25, 25, 25, 25, 25, 25, 25],
    "default_dof_pos": [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
    + [0.0, 0.0, 0.0]
    + [0.5, 0.0, 0.2, 0.3, 0.5, 0.0, -0.2, 0.3],
    "action_scale": 0.25,
    "ang_vel_scale": 0.25,
    "dof_vel_scale": 0.05,
    "gait_freq": 1.3,
    "num_actions": 15,
    "num_dofs": 23,
    "n_proprio": 93,
    "history_len": 10,
    "extra_history_len": 25,
    "dof_vel_zero_idx": [4, 5, 10, 11, 13, 14],  # ankles + waist roll/pitch
    "base_height": 0.75,
    "raw_action_clip": 40.0,
}

# in-distribution command ranges (README); commands outside degrade, they don't fail
AMO_COMMAND_RANGES: Dict[str, Tuple[float, float]] = {
    "vx": (-0.5, 0.5),
    "vy": (-0.4, 0.4),
    "height_delta": (-0.5, 0.8),
    "torso_yaw": (-1.57, 1.57),
    "torso_pitch": (-0.52, 1.57),
    "torso_roll": (-0.7, 0.7),
}


# --------------------------------------------------------------------------- weight loading
def load_torch_numpy_pickle(path: Path) -> Dict[str, np.ndarray]:
    """Read a ``torch.save``'d dict of *numpy arrays* without torch (restricted unpickler)."""
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0] + "/"
    allowed = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("_codecs", "encode"),
    }

    class Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if (module, name) in allowed:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")

        def persistent_load(self, pid: Any):  # torch storages never appear in these files
            raise pickle.UnpicklingError("unexpected torch storage in numpy-only pickle")

    return Unpickler(io.BytesIO(z.read(prefix + "data.pkl"))).load()


class AmoParams(NamedTuple):
    """All AMO weights as JAX arrays (policy + adapter + adapter normalisation stats)."""

    policy: Dict[str, Array]  # tensors of amo_jit.pt (dotted names)
    adapter: Dict[str, Array]  # tensors of adapter_jit.pt
    input_mean: Array  # (12,)
    input_std: Array  # (12,)
    output_mean: Array  # (15,)
    output_std: Array  # (15,)


def load_amo_params(offline: bool = False) -> AmoParams:
    d = amo_dir(offline=offline)
    pol = {k: jnp.asarray(v) for k, v in load_torchscript_tensors(d / "amo_jit.pt").items()}
    ada = {k: jnp.asarray(v) for k, v in load_torchscript_tensors(d / "adapter_jit.pt").items()}
    ns = load_torch_numpy_pickle(d / "adapter_norm_stats.pt")
    return AmoParams(
        policy=pol,
        adapter=ada,
        input_mean=jnp.asarray(ns["input_mean"], dtype=jnp.float32),
        input_std=jnp.asarray(ns["input_std"], dtype=jnp.float32),
        output_mean=jnp.asarray(ns["output_mean"], dtype=jnp.float32),
        output_std=jnp.asarray(ns["output_std"], dtype=jnp.float32),
    )


# --------------------------------------------------------------------------- network forwards
def _elu(x: Array) -> Array:
    return jax.nn.elu(x)


def _linear(p: Dict[str, Array], name: str, x: Array) -> Array:
    return x @ p[f"{name}.weight"].T + p[f"{name}.bias"]


def adapter_forward(params: AmoParams, x12: Array) -> Array:
    """The AMO adapter MLP: normalised ``[height, torso ypr, arm q(8)]`` -> 15-dim reference.

    Linear/BatchNorm1d(eval)/LeakyReLU x3 + Linear; input/output de/normalised with the
    stored stats (matches ``play_amo.py``).
    """
    p = params.adapter
    x = (jnp.asarray(x12, dtype=jnp.float32) - params.input_mean) / (params.input_std + 1e-8)

    def bn(i: int, v: Array) -> Array:
        mean, var = p[f"model.{i}.running_mean"], p[f"model.{i}.running_var"]
        return (v - mean) / jnp.sqrt(var + 1e-5) * p[f"model.{i}.weight"] + p[f"model.{i}.bias"]

    x = jax.nn.leaky_relu(bn(1, _linear(p, "model.0", x)), 0.01)
    x = jax.nn.leaky_relu(bn(4, _linear(p, "model.3", x)), 0.01)
    x = jax.nn.leaky_relu(bn(7, _linear(p, "model.6", x)), 0.01)
    x = _linear(p, "model.9", x)
    return x * params.output_std + params.output_mean


def _conv1d(x: Array, w: Array, b: Array, stride: int) -> Array:
    """(C_in, L) x (C_out, C_in, K) -> (C_out, L_out), valid padding."""
    out = jax.lax.conv_general_dilated(
        x[None],
        w,
        window_strides=(stride,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
    )[0]
    return out + b[:, None]


def _history_encoder(p: Dict[str, Array], hist: Array) -> Array:
    """(10, 93) history -> (20,): per-frame Linear+ELU, Conv1d(k4,s2)+ELU, Conv1d(k2,s1)+ELU,
    flatten, Linear+ELU. Strides (2, 1) verified against torch to 3e-8."""
    g = lambda k: p["_orig_actor.history_encoder." + k]  # noqa: E731
    e = _elu(hist @ g("encoder.0.weight").T + g("encoder.0.bias"))  # (10, 30)
    h = _elu(_conv1d(e.T, g("conv_layers.0.weight"), g("conv_layers.0.bias"), 2))  # (20, 4)
    h = _elu(_conv1d(h, g("conv_layers.2.weight"), g("conv_layers.2.bias"), 1))  # (10, 3)
    return _elu(h.reshape(-1) @ g("linear_output.0.weight").T + g("linear_output.0.bias"))


def policy_forward(
    params: AmoParams, obs_prop: Array, obs_demo: Array, hist: Array, extra_hist: Array
) -> Array:
    """The AMO student policy: (93,), (17,), (10, 93) history (excl. current frame),
    (25, 93) extra history (incl. current frame) -> 15 raw actions (legs + waist)."""
    p = params.policy
    f32 = jnp.float32
    obs_prop = jnp.asarray(obs_prop, dtype=f32)
    obs_demo = jnp.asarray(obs_demo, dtype=f32)
    hist = jnp.asarray(hist, dtype=f32)
    extra_hist = jnp.asarray(extra_hist, dtype=f32)
    tfe = _elu(
        _elu(
            hist[-4:] @ p["_orig_actor.text_feat_encoder.0.weight"].T
            + p["_orig_actor.text_feat_encoder.0.bias"]
        )
        @ p["_orig_actor.text_feat_encoder.2.weight"].T
        + p["_orig_actor.text_feat_encoder.2.bias"]
    )  # (4, 16)
    feat = _elu(
        tfe.reshape(-1) @ p["_orig_actor.text_feat_merger.0.weight"].T
        + p["_orig_actor.text_feat_merger.0.bias"]
    )  # (16,)
    hist_enc = _history_encoder(p, hist)  # (20,)
    x = jnp.concatenate(
        [extra_hist.reshape(-1), feat, obs_prop, obs_demo, jnp.zeros(3, dtype=f32), hist_enc]
    )  # 2325 + 16 + 93 + 17 + 3 + 20 = 2474
    x = _elu(_linear(p, "student_actor_backbone.0", x))
    x = _elu(_linear(p, "student_actor_backbone.2", x))
    x = _elu(_linear(p, "student_actor_backbone.4", x))
    return _linear(p, "student_actor_backbone.6", x)


# --------------------------------------------------------------------------- plant
def load_g1_23dof(offline: bool = False) -> mujoco.MjModel:
    """AMO's 23-DoF G1 (deployment-trimmed collisions: 8 foot spheres + pelvis vs floor).

    Upstream requests the PGS solver, which MJX does not implement -- switched to Newton.
    Timestep set to play_amo's 2 ms (the XML says 1 ms; ``play_amo.py`` overrides it too).
    """
    m = mujoco.MjModel.from_xml_path(str(amo_dir(offline=offline) / "g1.xml"))
    if (m.nq, m.nv, m.nu) != (30, 29, 23):
        raise RuntimeError(
            f"unexpected AMO g1 model dims {(m.nq, m.nv, m.nu)}; observation layout assumes (30, 29, 23)"
        )
    m.opt.timestep = AMO_CONFIG["simulation_dt"]
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    return m


def make_g1_23dof_plant(offline: bool = False) -> MujocoPlant:
    """``MujocoPlant`` for AMO's G1: ``u`` = 23 joint targets, PD torque at every 2 ms substep."""
    m = load_g1_23dof(offline=offline)
    kp = jnp.asarray(AMO_CONFIG["kps"], dtype=float)
    kd = jnp.asarray(AMO_CONFIG["kds"], dtype=float)
    lim = jnp.asarray(AMO_CONFIG["torque_limits"], dtype=float)

    def pd(data, target_q):
        tau = kp * (target_q - data.qpos[7:30]) - kd * data.qvel[6:29]
        return jnp.clip(tau, -lim, lim)

    return MujocoPlant(m, substeps=AMO_CONFIG["control_decimation"], ctrl_map=pd, nu=23)


def x0_standing(plant: MujocoPlant) -> Array:
    """play_amo's initial state: the XML ``home`` keyframe, zero velocity."""
    kq = jnp.asarray(plant.mj_model.key_qpos[0])
    return plant.to_state(plant.from_state(jnp.concatenate([kq, jnp.zeros(plant.nv + 3)])))


# --------------------------------------------------------------------------- controller
class AmoState(NamedTuple):
    """Carry of one AMO control step (rides in ``sub_data["_amo"]``)."""

    hist: Array  # (10, 93) proprio history, oldest first, EXCLUDING the current frame
    extra_hist: Array  # (25, 93) extra history, oldest first, INCLUDING the current frame
    last_action: Array  # (23,) = [15 raw actions | 8 arm displacements / action_scale]
    gait: Array  # (2,) gait phase in [0, 1)
    stand: Array  # () bool: previous step's in-place-stand flag (gates dyaw)
    target_yaw: Array  # () followed heading (kept while the command speed is ~0)


def _rpy(quat: Array) -> Array:
    """(roll, pitch, yaw) of a (w, x, y, z) quaternion -- play_amo's ``quatToEuler``."""
    qw, qx, qy, qz = quat
    roll = jnp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = jnp.arcsin(jnp.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = jnp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return jnp.array([roll, pitch, yaw])


TorsoCommand = Union[None, Tuple[float, float, float, float], Callable[[Array], Array]]


class AmoWholeBodyPolicy:
    """AMO's whole-body policy as a CBFKit controller factory. See the module docstring."""

    def __init__(self, params: Optional[AmoParams] = None, offline: bool = False) -> None:
        self.params = params if params is not None else load_amo_params(offline=offline)
        c = AMO_CONFIG
        self.q_default = jnp.asarray(c["default_dof_pos"], dtype=float)
        self.n_act = int(c["num_actions"])
        self.n_dof = int(c["num_dofs"])
        self.action_scale = float(c["action_scale"])
        self.gait_freq = float(c["gait_freq"])
        self.control_dt = float(c["simulation_dt"]) * int(c["control_decimation"])
        self._vel_zero = jnp.zeros(self.n_dof).at[jnp.asarray(c["dof_vel_zero_idx"])].set(1.0)

    def init_state(self) -> AmoState:
        c = AMO_CONFIG
        return AmoState(
            hist=jnp.zeros((c["history_len"], c["n_proprio"])),
            extra_hist=jnp.zeros((c["extra_history_len"], c["n_proprio"])),
            last_action=jnp.zeros(self.n_dof),
            gait=jnp.array([0.25, 0.25]),
            stand=jnp.asarray(True),
            target_yaw=jnp.zeros(()),
        )

    def step(self, x: Array, cmd: Array, state: AmoState) -> Tuple[Array, AmoState]:
        """One 50 Hz step. ``x`` = flat plant state ``[qpos(30) | qvel(29) | com]``;
        ``cmd`` = ``[vx, target_yaw, vy, height_delta, torso_yaw, torso_pitch, torso_roll]``
        (play_amo's ``viewer.commands`` order). Returns 23 joint PD targets + new carry."""
        c = AMO_CONFIG
        cmd = jnp.asarray(cmd, dtype=float)
        dof_pos = x[7:30]
        dof_vel = x[30 + 6 : 30 + 29]
        quat = x[3:7]
        ang_vel = x[30 + 3 : 30 + 6]
        rpy = _rpy(quat)

        dyaw = _wrap(rpy[2] - cmd[1])
        dyaw = jnp.where(state.stand, 0.0, dyaw)  # previous step's flag, as in play_amo
        obs_dof_vel = dof_vel * (1.0 - self._vel_zero)

        adapter_in = jnp.concatenate(
            [jnp.array([c["base_height"] + cmd[3], cmd[4], cmd[5], cmd[6]]), dof_pos[15:]]
        )
        adapter_out = adapter_forward(self.params, adapter_in)

        obs_prop = jnp.concatenate(
            [
                ang_vel * c["ang_vel_scale"],
                rpy[:2],
                jnp.array([jnp.sin(dyaw), jnp.cos(dyaw)]),
                dof_pos - self.q_default,
                obs_dof_vel * c["dof_vel_scale"],
                state.last_action,
                jnp.sin(state.gait * 2 * jnp.pi),
                adapter_out,
            ]
        ).astype(jnp.float32)

        stand_new = jnp.abs(cmd[0]) < 0.1
        obs_demo = jnp.concatenate(
            [
                dof_pos[15:],
                jnp.array([cmd[0], cmd[2], 0.0, cmd[4], cmd[5], cmd[6]]),
                jnp.full(3, c["base_height"] + cmd[3]),
            ]
        ).astype(jnp.float32)

        extra_new = jnp.roll(state.extra_hist, -1, axis=0).at[-1].set(obs_prop)
        raw = policy_forward(self.params, obs_prop, obs_demo, state.hist, extra_new)
        raw = jnp.clip(raw, -c["raw_action_clip"], c["raw_action_clip"])

        pd_target = jnp.concatenate(
            [raw * self.action_scale + self.q_default[:15], self.q_default[15:]]
        )
        last_action = jnp.concatenate([raw, (dof_pos - self.q_default)[15:] / self.action_scale])

        gait = jnp.remainder(state.gait + self.control_dt * self.gait_freq, 1.0)
        near = jnp.abs(gait - 0.25) < 0.05
        gait = jnp.where(stand_new & (near[0] | near[1]), jnp.array([0.25, 0.25]), gait)
        gait = jnp.where(~stand_new & near[0] & near[1], jnp.array([0.25, 0.75]), gait)

        new_state = AmoState(
            hist=jnp.roll(state.hist, -1, axis=0).at[-1].set(obs_prop),
            extra_hist=extra_new,
            last_action=last_action,
            gait=gait,
            stand=stand_new,
            target_yaw=state.target_yaw,
        )
        return pd_target, new_state

    def as_controller(
        self,
        *,
        torso_command: TorsoCommand = None,
        world_frame: bool = True,
        min_speed_for_heading: float = 0.1,
    ):
        """``(t, x, u_nom, key, data) -> (q_target(23), data)``.

        ``u_nom[:2]`` is a planar velocity command -- world-frame by default (what the
        CoM-level CBF layer emits). Heading following is *native* to AMO: the command
        carries an absolute ``target_yaw``, set here to the command's direction (held at
        its last value below ``min_speed_for_heading``, where AMO stands in place); the
        speed is sent as body-frame ``(vx, vy)``, so the commanded world velocity is
        realised while the robot turns. ``torso_command`` adds the whole-body part:
        ``None`` (upright), a constant ``(height_delta, torso_yaw, torso_pitch,
        torso_roll)``, or a callable ``t -> (4,)`` for scheduled motions. Commands are
        clipped to the in-distribution ranges (``AMO_COMMAND_RANGES``). The carry lives in
        ``sub_data["_amo"]``; the assembled 7-command is logged as ``amo_cmd``.
        """
        if torso_command is None:
            torso_fn = lambda t: jnp.zeros(4)  # noqa: E731
        elif callable(torso_command):
            torso_fn = torso_command
        else:
            const = jnp.asarray(torso_command, dtype=float)
            torso_fn = lambda t: const  # noqa: E731
        r = AMO_COMMAND_RANGES

        def controller(t, x, u_nom, key, data):
            sub = dict(data.sub_data) if data.sub_data is not None else {}
            state = sub.get("_amo")
            if state is None:
                state = self.init_state()
            v = jnp.asarray(u_nom, dtype=float)[:2]
            yaw = _yaw(x[3:7])
            speed = jnp.linalg.norm(v)
            target_yaw = jnp.where(
                speed > min_speed_for_heading, jnp.arctan2(v[1], v[0]), state.target_yaw
            )
            if world_frame:
                cs, sn = jnp.cos(yaw), jnp.sin(yaw)
                v_body = jnp.array([cs * v[0] + sn * v[1], -sn * v[0] + cs * v[1]])
            else:
                v_body = v
                target_yaw = jnp.where(
                    speed > min_speed_for_heading, yaw + jnp.arctan2(v[1], v[0]), state.target_yaw
                )
            torso = jnp.asarray(torso_fn(t), dtype=float)
            cmd = jnp.array(
                [
                    jnp.clip(v_body[0], *r["vx"]),
                    target_yaw,
                    jnp.clip(v_body[1], *r["vy"]),
                    jnp.clip(torso[0], *r["height_delta"]),
                    jnp.clip(torso[1], *r["torso_yaw"]),
                    jnp.clip(torso[2], *r["torso_pitch"]),
                    jnp.clip(torso[3], *r["torso_roll"]),
                ]
            )
            u, state = self.step(x, cmd, state._replace(target_yaw=target_yaw))
            sub["_amo"] = state
            sub["amo_cmd"] = cmd
            return u, data._replace(sub_data=sub, u=u, u_nom=jnp.asarray(u_nom, dtype=float))

        controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
        return controller


__all__ = [
    "AMO_COMMAND_RANGES",
    "AMO_CONFIG",
    "AmoParams",
    "AmoState",
    "AmoWholeBodyPolicy",
    "adapter_forward",
    "load_amo_params",
    "load_g1_23dof",
    "load_torch_numpy_pickle",
    "make_g1_23dof_plant",
    "policy_forward",
    "x0_standing",
]
