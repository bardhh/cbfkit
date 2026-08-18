"""``execute(plant=...)`` on the vendored contact-free cart-pole."""

import jax
import jax.numpy as jnp
import pytest

import cbfkit.simulation.simulator as sim
from cbfkit.systems.mujoco import load_model
from cbfkit.systems.mujoco.plant import MujocoPlant
from cbfkit.utils.user_types import ControllerData

X0 = jnp.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
N = 30


@pytest.fixture(scope="module")
def plant():
    return MujocoPlant(load_model("cart_pole"))


def _pd_controller(t, x, u_nom, key, data):
    # Push the cart toward x=0 and damp; 5-arg canonical signature.
    u = jnp.atleast_1d(-2.0 * x[0] - 0.5 * x[2])
    return u, data


def _manual_rollout(plant, controller, n):
    """Ground truth: what the simulator's JIT path should log (pre-step states)."""
    d = plant.from_state(X0)
    xs, us = [], []
    for k in range(n):
        x = plant.to_state(d)
        u, _ = controller(k * plant.dt, x, jnp.zeros(1), None, ControllerData())
        xs.append(x)
        us.append(u)
        d = plant.step(d, u)
    return jnp.stack(xs), jnp.stack(us)


def test_jit_plant_path_matches_manual_rollout(plant):
    res = sim.execute(
        x0=X0,
        dt=plant.dt,
        num_steps=N,
        plant=plant,
        controller=_pd_controller,
        use_jit=True,
        verbose=False,
    )
    xs_ref, us_ref = _manual_rollout(plant, _pd_controller, N)
    assert res.states.shape == (N, plant.state_dim)
    assert res.controls.shape == (N, plant.nu)
    assert jnp.allclose(res.states, xs_ref, atol=1e-9)
    assert jnp.allclose(res.controls, us_ref, atol=1e-9)
    # Legacy 8-tuple unpacking still works.
    x, u, z, p, dk, dv, pk, pv = res
    assert x.shape == (N, plant.state_dim)


def test_jit_plant_path_zero_controller_and_estimates(plant):
    res = sim.execute(
        x0=X0,
        dt=plant.dt,
        num_steps=5,
        plant=plant,
        use_jit=True,
        verbose=False,
        controller=lambda t, x, u_nom, key, data: (jnp.zeros(1), data),
    )
    assert res.estimates.shape == (5, plant.state_dim)
    assert res.covariances.shape == (5, plant.state_dim, plant.state_dim)


def test_jit_plant_path_nan_guard_freezes_state(plant):
    def _nan_after_3(t, x, u_nom, key, data):
        u = jnp.where(t > 2.5 * plant.dt, jnp.nan, 0.0)
        return jnp.atleast_1d(u), data

    res = sim.execute(
        x0=X0,
        dt=plant.dt,
        num_steps=10,
        plant=plant,
        controller=_nan_after_3,
        use_jit=True,
        verbose=False,
    )
    assert jnp.all(jnp.isfinite(res.states))
    # After the NaN the held state repeats.
    assert jnp.allclose(res.states[-1], res.states[-2])


def test_eager_plant_path_matches_jit_shifted_by_one(plant):
    # Eager logs post-step x_{k+1}; JIT logs pre-step x_k (same convention as
    # tests/test_simulation/test_backend_parity.py).
    with pytest.warns(UserWarning, match="debug-only"):
        res_py = sim.execute(
            x0=X0,
            dt=plant.dt,
            num_steps=N,
            plant=plant,
            controller=_pd_controller,
            use_jit=False,
            verbose=False,
        )
    res_jit = sim.execute(
        x0=X0,
        dt=plant.dt,
        num_steps=N,
        plant=plant,
        controller=_pd_controller,
        use_jit=True,
        verbose=False,
    )
    assert jnp.allclose(res_py.states[:-1], res_jit.states[1:], atol=1e-9)
    assert jnp.allclose(res_py.controls, res_jit.controls, atol=1e-9)


def test_eager_and_jit_plant_paths_share_rng_stream(plant):
    # A controller that leaks its key into the control proves the split order is pinned.
    def _keyed(t, x, u_nom, key, data):
        return jnp.atleast_1d(0.01 * jax.random.normal(key)), data

    with pytest.warns(UserWarning):
        res_py = sim.execute(
            x0=X0,
            dt=plant.dt,
            num_steps=8,
            plant=plant,
            controller=_keyed,
            use_jit=False,
            verbose=False,
            key=jax.random.PRNGKey(3),
        )
    res_jit = sim.execute(
        x0=X0,
        dt=plant.dt,
        num_steps=8,
        plant=plant,
        controller=_keyed,
        use_jit=True,
        verbose=False,
        key=jax.random.PRNGKey(3),
    )
    assert jnp.allclose(res_py.controls, res_jit.controls, atol=1e-12)
