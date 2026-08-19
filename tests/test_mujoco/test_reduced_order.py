"""Reduced-order CBF on the plant's CoM: embedded dynamics, barriers, and the safe-locomotion wrapper."""

import urllib.error

import jax
import jax.numpy as jnp
import pytest

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.systems.mujoco.plant import MujocoPlant
from cbfkit.systems.mujoco.reduced_order import (
    com_obstacle_barriers,
    embedded_single_integrator,
    safe_locomotion_controller,
)
from cbfkit.utils.user_types import ControllerData


@pytest.fixture(scope="module")
def g1_plant():
    g1 = pytest.importorskip("cbfkit.systems.mujoco.g1")
    try:
        return MujocoPlant(g1.load_g1()), g1
    except (RuntimeError, urllib.error.URLError) as exc:
        pytest.skip(f"G1 assets unavailable: {exc}")


def test_embedded_single_integrator_shapes(g1_plant):
    plant, _ = g1_plant
    f, g = embedded_single_integrator(plant.state_dim, plant.com_indices)(
        jnp.zeros(plant.state_dim)
    )
    assert f.shape == (plant.state_dim,) and g.shape == (plant.state_dim, 2)
    assert float(g[plant.com_indices[0], 0]) == 1.0 and float(g[plant.com_indices[1], 1]) == 1.0
    assert float(jnp.abs(g).sum()) == 2.0  # nothing else


def test_barrier_is_evaluated_at_the_com_not_the_pelvis(g1_plant):
    plant, g1mod = g1_plant
    g1 = g1mod.G1(plant.mj_model)
    x = g1.x_stand(plant)
    com = x[plant.com_indices[0] : plant.com_indices[0] + 2]
    # Obstacle centred exactly on the CoM -> h = -1 there; centred on the pelvis xy -> h != -1
    # unless CoM == pelvis (it isn't: measured offset at stand).
    b_com = com_obstacle_barriers(plant, [com], [(0.3, 0.3)])
    b_pel = com_obstacle_barriers(plant, [x[0:2]], [(0.3, 0.3)])
    assert float(b_com.functions[0](0.0, x)) == pytest.approx(-1.0)
    assert float(b_pel.functions[0](0.0, x)) != pytest.approx(-1.0)


def test_cbf_filters_command_away_from_obstacle(g1_plant):
    plant, g1mod = g1_plant
    g1 = g1mod.G1(plant.mj_model)
    x = g1.x_stand(plant)
    com = x[plant.com_indices[0] : plant.com_indices[0] + 2]
    obstacle = com + jnp.array([0.6, 0.0])  # just ahead, ellipsoid radius 0.5 -> h = 0.44
    dyn = embedded_single_integrator(plant.state_dim, plant.com_indices)
    barriers = com_obstacle_barriers(plant, [obstacle], [(0.5, 0.5)], class_k_gain=1.0)
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0]), dynamics_func=dyn, barriers=barriers
    )
    v_nom = jnp.array([1.0, 0.0])  # straight into the obstacle
    v_safe, data = cbf_qp(0.0, x, v_nom, jax.random.PRNGKey(0), ControllerData())
    assert not bool(data.error)
    assert float(v_safe[0]) < float(v_nom[0])  # slowed toward the obstacle
    # Constraint check: Lgh v + alpha h >= 0 with h = ((cx-ox)/a)^2 + ((cy-oy)/b)^2 - 1
    d = com - obstacle
    h = float((d[0] / 0.5) ** 2 + (d[1] / 0.5) ** 2 - 1)
    grad = 2 * d / 0.5**2
    assert float(grad @ v_safe) + 1.0 * h >= -1e-6


def test_safe_locomotion_wrapper_composes_and_logs(g1_plant):
    plant, g1mod = g1_plant
    g1 = g1mod.G1(plant.mj_model)
    x = g1.x_stand(plant)
    dyn = embedded_single_integrator(plant.state_dim, plant.com_indices)
    barriers = com_obstacle_barriers(
        plant, [x[plant.com_indices[0] : plant.com_indices[0] + 2] + 5.0], [(0.5, 0.5)]
    )
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0]), dynamics_func=dyn, barriers=barriers
    )
    seen = {}

    def fake_locomotion(t, xx, cmd, key, data):  # stands in for SamplingMpc.as_controller()
        seen["cmd"] = cmd
        sub = dict(data.sub_data or {})
        sub["_loco"] = jnp.asarray(sub.get("_loco", 0.0)) + 1.0  # carry-only state
        return jnp.zeros(plant.nu), data._replace(sub_data=sub)

    ctrl = safe_locomotion_controller(cbf_qp, fake_locomotion)
    u, d1 = ctrl(0.0, x, jnp.array([0.4, 0.1]), jax.random.PRNGKey(0), ControllerData())
    u, d2 = ctrl(0.02, x, jnp.array([0.4, 0.1]), jax.random.PRNGKey(1), d1)
    assert u.shape == (plant.nu,)
    assert jnp.allclose(seen["cmd"], d2.sub_data["v_safe"])
    assert jnp.allclose(d2.sub_data["v_nom"], jnp.array([0.4, 0.1]))
    assert (
        float(d2.sub_data["_loco"]) == 2.0
    )  # carry-only state survived the CBF's sub_data rewrite
    assert not bool(d2.error)


def test_double_integrator_hocbf_brakes_toward_obstacle(g1_plant):
    from cbfkit.systems.mujoco.reduced_order import (
        com_obstacle_hocbfs,
        embedded_double_integrator,
        safe_locomotion_controller_di,
    )

    plant, g1mod = g1_plant
    g1 = g1mod.G1(plant.mj_model)
    x = g1.x_stand(plant)
    com = x[plant.com_indices[0] : plant.com_indices[0] + 2]
    obstacle = com + jnp.array([0.9, 0.0])  # ahead; keep-out radius 0.6 -> h = 1.25
    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices)
    f, g = dyn(jnp.concatenate([x, jnp.array([0.3, 0.1])]))
    assert f.shape == (plant.state_dim + 2,) and g.shape == (plant.state_dim + 2, 2)
    assert float(f[plant.com_indices[0]]) == pytest.approx(0.3)  # d/dt com = v
    barriers = com_obstacle_hocbfs(plant, [obstacle], [(0.6, 0.6)], class_k_gain=1.0)
    cbf_qp = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([2.0, 2.0]), dynamics_func=dyn, barriers=barriers
    )
    # Already moving fast at the obstacle: nominal wants to hold 0.5 m/s; the HOCBF must brake.
    xa = jnp.concatenate([x, jnp.array([0.5, 0.0])])
    a_safe, data = cbf_qp(0.0, xa, jnp.array([0.0, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert not bool(data.error)
    assert float(a_safe[0]) < -0.05  # decelerating along x
    # Through the wrapper: integrated command shrinks toward the obstacle and is carried in _di_v.
    seen = {}

    def fake_locomotion(t, xx, cmd, key, d):
        seen["cmd"] = cmd
        return jnp.zeros(plant.nu), d

    ctrl = safe_locomotion_controller_di(cbf_qp, fake_locomotion, plant, 0.02, v_max=0.5)
    d = ControllerData(sub_data={"_di_v": jnp.array([0.5, 0.0])})
    _, d1 = ctrl(0.0, x, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0), d)
    assert float(d1.sub_data["_di_v"][0]) < 0.5
    assert jnp.allclose(seen["cmd"], d1.sub_data["v_safe"])
    assert "a_safe" in d1.sub_data
