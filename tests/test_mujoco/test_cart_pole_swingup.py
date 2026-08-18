"""Milestone 1 acceptance: MJX sampling MPC swings the cart-pole up, driven by execute(plant=...)."""

import jax.numpy as jnp
import pytest

import cbfkit.simulation.simulator as sim
from cbfkit.controllers.mjx_sampling_mpc import SamplingMpc
from cbfkit.systems.mujoco import load_model
from cbfkit.systems.mujoco.plant import MujocoPlant


def _upright_dist(qpos):
    theta = qpos[1] + jnp.pi
    return (jnp.cos(theta) - 1.0) ** 2 + jnp.sin(theta) ** 2


def running_cost(data, u, aux):
    return (
        _upright_dist(data.qpos)
        + data.qpos[0] ** 2
        + 0.01 * jnp.sum(data.qvel**2)
        + 0.01 * jnp.sum(u**2)
    )


def terminal_cost(data, aux):
    return 10.0 * _upright_dist(data.qpos) + data.qpos[0] ** 2 + 0.01 * jnp.sum(data.qvel**2)


@pytest.mark.slow
def test_cart_pole_swingup_through_execute():
    plant = MujocoPlant(load_model("cart_pole"), substeps=2)  # control at 50 Hz
    mpc = SamplingMpc(
        plant,
        running_cost,
        terminal_cost,
        num_samples=128,
        plan_horizon=1.0,
        noise_level=0.3,
        temperature=0.1,
        num_knots=4,
        spline_type="linear",
    )
    controller = mpc.as_controller()
    x0 = jnp.zeros(plant.state_dim)  # pole hanging down (theta=0 is down in this model)
    steps = 200  # 4 s
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        plant=plant,
        controller=controller,
        use_jit=True,
        verbose=False,
    )
    assert res.states.shape == (steps, plant.state_dim)
    dist = jnp.array([_upright_dist(q) for q in res.states[:, :2]])
    assert float(dist[0]) > 3.5  # starts hanging down
    assert float(jnp.min(dist[-50:])) < 0.5  # within ~30 deg of upright in the last second
    assert float(jnp.max(jnp.abs(res.states[:, 0]))) < 1.8  # stays on the rail
    # MPC state was carried, not logged as an array column.
    assert not any(k.startswith("sub_data__mpc") for k in res.controller_keys)
