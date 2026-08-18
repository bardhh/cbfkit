"""Unitree G1 helper: loader, ids/keyframes, standup costs, friction DR."""

import urllib.error

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from cbfkit.systems.mujoco.plant import MujocoPlant


@pytest.fixture(scope="module")
def g1mod():
    g1 = pytest.importorskip("cbfkit.systems.mujoco.g1")
    try:
        g1.load_g1()
    except (RuntimeError, urllib.error.URLError) as exc:
        pytest.skip(f"G1 assets unavailable: {exc}")
    return g1


def test_load_g1_dimensions_and_sim_overrides(g1mod):
    m = g1mod.load_g1()
    assert (m.nq, m.nv, m.nu) == (36, 35, 29)
    assert m.opt.timestep == pytest.approx(0.02)
    ms = g1mod.load_g1(sim=True)
    assert ms.opt.timestep == pytest.approx(0.01)
    assert np.allclose(ms.geom_solimp, [0.9, 0.95, 0.001, 0.5, 2])
    assert not (ms.opt.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE)  # MJX rejects it


def test_g1_ids_and_keyframes(g1mod):
    m = g1mod.load_g1()
    plant = MujocoPlant(m)
    g1 = g1mod.G1(m)
    assert g1.q_stand.shape == (36,)
    x_stand = g1.x_stand(plant)
    assert x_stand.shape == (plant.state_dim,)
    d = plant.from_state(x_stand)
    assert 0.8 < float(g1.torso_height(d)) < 1.05
    assert float(g1.torso_upright(d)) > 0.95
    d_f = plant.from_state(g1.x_fallen(plant))
    assert float(g1.torso_upright(d_f)) < 0.5  # lying on its side


def test_standup_costs_prefer_standing(g1mod):
    m = g1mod.load_g1()
    plant = MujocoPlant(m)
    g1 = g1mod.G1(m)
    running, terminal = g1mod.standup_costs(g1)
    u0 = jnp.zeros(plant.nu)
    d_s = plant.from_state(g1.x_stand(plant))
    d_f = plant.from_state(g1.x_fallen(plant))
    for d in (d_s, d_f):
        assert jnp.isfinite(running(d, u0, None)) and jnp.isfinite(terminal(d, None))
    assert float(running(d_s, u0, None)) < float(running(d_f, u0, None))
    assert float(terminal(d_s, None)) < float(terminal(d_f, None))


def test_friction_randomizer_shapes(g1mod):
    m = g1mod.load_g1()
    plant = MujocoPlant(m)
    r = g1mod.friction_randomizer(0.5, 2.0)
    out = r(plant.model, jax.random.PRNGKey(0))
    assert set(out) == {"geom_friction"}
    assert out["geom_friction"].shape == plant.model.geom_friction.shape
    ratio = out["geom_friction"][:, 0] / plant.model.geom_friction[:, 0]
    assert jnp.all((ratio >= 0.5) & (ratio <= 2.0))
    # Steps under the randomised model.
    d = plant.step(plant.make_data(), jnp.zeros(plant.nu), model=plant.model.tree_replace(out))
    assert isinstance(d, mjx.Data)
