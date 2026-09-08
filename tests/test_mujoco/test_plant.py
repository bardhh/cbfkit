import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from cbfkit.systems.mujoco import load_model
from cbfkit.systems.mujoco.plant import MujocoPlant


@pytest.fixture(scope="module")
def plant():
    return MujocoPlant(load_model("cart_pole"))


def test_dimensions(plant):
    assert (plant.nq, plant.nv, plant.nu) == (2, 2, 1)
    assert plant.state_dim == 2 + 2 + 3
    assert plant.com_indices == (4, 5)
    assert plant.dt == pytest.approx(0.01)
    assert plant.u_min.shape == (1,) and plant.u_max.shape == (1,)
    assert float(plant.u_min[0]) == -1.0 and float(plant.u_max[0]) == 1.0


def test_substeps_scale_dt():
    p = MujocoPlant(load_model("cart_pole"), substeps=2)
    assert p.dt == pytest.approx(0.02)


def test_to_state_layout_and_com(plant):
    d = plant.make_data()
    x = plant.to_state(d)
    assert x.shape == (plant.state_dim,)
    assert jnp.allclose(x[: plant.nq], d.qpos)
    assert jnp.allclose(x[plant.nq : plant.nq + plant.nv], d.qvel)
    # CoM must agree with CPU MuJoCo's subtree_com of the world body.
    md = mujoco.MjData(plant.mj_model)
    mujoco.mj_forward(plant.mj_model, md)
    assert np.allclose(np.asarray(x[plant.nq + plant.nv :]), md.subtree_com[0], atol=1e-9)


def test_to_state_com_tracks_current_qpos_not_stale_data(plant):
    # After a step, mjx.Data.subtree_com refers to the pre-integration qpos.
    # to_state must recompute the CoM from the integrated qpos.
    d = plant.make_data().replace(qpos=jnp.array([0.5, 0.3]), qvel=jnp.array([2.0, 0.0]))
    d = mjx.forward(plant.model, d)
    d1 = plant.step(d, jnp.zeros(1))
    md = mujoco.MjData(plant.mj_model)
    md.qpos[:] = np.asarray(d1.qpos)
    md.qvel[:] = np.asarray(d1.qvel)
    mujoco.mj_forward(plant.mj_model, md)
    x1 = plant.to_state(d1)
    assert np.allclose(np.asarray(x1[plant.nq + plant.nv :]), md.subtree_com[0], atol=1e-8)


def test_from_state_round_trip(plant):
    x = jnp.array([0.4, -1.0, 0.1, 0.2, 0.0, 0.0, 0.0])
    d = plant.from_state(x)
    x2 = plant.to_state(d)
    assert jnp.allclose(x2[:4], x[:4])  # qpos|qvel exact
    assert jnp.all(jnp.isfinite(x2[4:]))  # CoM recomputed, tail of x ignored


def test_step_matches_cpu_mujoco_contact_free(plant):
    # cart_pole disables contact, so MJX and C MuJoCo agree tightly.
    d = plant.from_state(jnp.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    md = mujoco.MjData(plant.mj_model)
    md.qpos[:] = [0.0, 0.5]
    u = jnp.array([0.7])
    step = jax.jit(plant.step)
    for _ in range(50):
        d = step(d, u)
        md.ctrl[:] = np.asarray(u)
        mujoco.mj_step(plant.mj_model, md)
    assert np.allclose(np.asarray(d.qpos), md.qpos, atol=1e-6)
    assert np.allclose(np.asarray(d.qvel), md.qvel, atol=1e-5)


def test_step_with_substeps_equals_repeated_single_steps():
    p1 = MujocoPlant(load_model("cart_pole"), substeps=1)
    p3 = MujocoPlant(load_model("cart_pole"), substeps=3)
    x0 = jnp.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    d1 = p1.from_state(x0)
    d3 = p3.from_state(x0)
    u = jnp.array([-0.4])
    for _ in range(3):
        d1 = p1.step(d1, u)
    d3 = p3.step(d3, u)
    assert jnp.allclose(d1.qpos, d3.qpos, atol=1e-10)
    assert jnp.allclose(d1.qvel, d3.qvel, atol=1e-10)


def test_step_accepts_alternate_model(plant):
    heavier = plant.model.tree_replace({"body_mass": plant.model.body_mass * 2.0})
    d = plant.make_data()
    u = jnp.array([1.0])
    d_a = plant.step(d, u)
    d_b = plant.step(d, u, model=heavier)
    assert not jnp.allclose(d_a.qvel, d_b.qvel)


def test_body_helpers(plant):
    bid = plant.body_id("cart")
    d = plant.make_data()
    p = plant.body_pos(d, bid)
    assert p.shape == (3,)
    assert float(p[2]) == pytest.approx(1.0)  # cart body sits at z=1 in the model
