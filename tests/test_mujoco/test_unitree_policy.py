"""Unitree's pretrained G1 walking policy: torch-free loading, JAX evaluation, PD plant, walking."""

import urllib.error

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cbfkit.simulation.simulator as sim
from cbfkit.utils.user_types import ControllerData


@pytest.fixture(scope="module")
def up():
    mod = pytest.importorskip("cbfkit.systems.mujoco.unitree_policy")
    try:
        mod.unitree_rl_gym_dir()
    except (RuntimeError, urllib.error.URLError) as exc:
        pytest.skip(f"unitree_rl_gym assets unavailable: {exc}")
    return mod


def test_torchscript_loader_recovers_lstm_and_actor(up):
    params = up.load_g1_policy_params()
    assert params.w_ih.shape == (256, 47) and params.w_hh.shape == (256, 64)
    assert params.b_ih.shape == (256,) and params.b_hh.shape == (256,)
    assert params.w1.shape == (32, 64) and params.w2.shape == (12, 32)
    assert all(bool(jnp.all(jnp.isfinite(p))) for p in params)


def test_plant_dims_and_pd_map(up):
    plant = up.make_g1_12dof_plant()
    assert (plant.nq, plant.nv, plant.nu) == (19, 18, 12)
    assert plant.dt == pytest.approx(0.02)  # 2 ms x 10 substeps
    d = plant.make_data()
    q_target = jnp.asarray(up.G1_12DOF_CONFIG["default_angles"])
    d1 = plant.step(d, q_target)
    # PD pulls the joints toward the target within a control step.
    err0 = float(jnp.linalg.norm(d.qpos[7:19] - q_target))
    err1 = float(jnp.linalg.norm(d1.qpos[7:19] - q_target))
    assert err1 < err0


def test_policy_step_shapes_and_state_carry(up):
    plant = up.make_g1_12dof_plant()
    pol = up.UnitreeG1WalkPolicy()
    x = up.x0_standing(plant)
    s0 = pol.init_state()
    obs = pol.observation(x, jnp.array([0.5, 0.0, 0.0]), 0.0, s0.last_action)
    assert obs.shape == (47,)
    u, s1 = pol.step(x, jnp.array([0.5, 0.0, 0.0]), 0.0, s0)
    assert u.shape == (12,) and s1.h.shape == (64,)
    assert not jnp.allclose(s1.h, s0.h)  # LSTM state advanced
    # as_controller carries state in sub_data["_policy"] and is memoised.
    ctrl = pol.as_controller()
    assert ctrl is pol.as_controller()
    u2, d = ctrl(0.0, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert "_policy" in d.sub_data and jnp.allclose(u2, u)


@pytest.mark.slow
def test_policy_walks_forward_in_mjx(up):
    plant = up.make_g1_12dof_plant()
    ctrl = up.UnitreeG1WalkPolicy().as_controller()

    def nominal(t, x, key, ref):
        return jnp.array([0.5, 0.0]), ControllerData()

    res = sim.execute(
        x0=up.x0_standing(plant),
        dt=plant.dt,
        num_steps=150,
        plant=plant,
        nominal_controller=nominal,
        controller=ctrl,
        use_jit=True,
        verbose=False,
    )
    S = np.asarray(res["states"])
    vx = S[50:, 19]  # after 1 s of transient
    assert S[:, 2].min() > 0.6  # never falls (pelvis stays up)
    assert 0.35 < vx.mean() < 0.65  # tracks 0.5 m/s
    assert S[-1, 0] - S[0, 0] > 1.0  # actually travelled
