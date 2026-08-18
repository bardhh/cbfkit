import jax
import jax.numpy as jnp
import pytest

from cbfkit.controllers.mjx_sampling_mpc import MpcState, SamplingMpc
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
    return 10.0 * _upright_dist(data.qpos) + data.qpos[0] ** 2


@pytest.fixture(scope="module")
def plant():
    return MujocoPlant(load_model("cart_pole"), substeps=2)  # dt = 0.02


@pytest.fixture(scope="module")
def mpc(plant):
    return SamplingMpc(
        plant,
        running_cost,
        terminal_cost,
        num_samples=32,
        plan_horizon=0.6,
        noise_level=0.3,
        temperature=0.1,
        num_knots=4,
        spline_type="linear",
    )


def test_init_state_shapes(mpc, plant):
    s = mpc.init_state()
    assert isinstance(s, MpcState)
    assert s.tk.shape == (4,) and float(s.tk[0]) == 0.0 and float(s.tk[-1]) == pytest.approx(0.6)
    assert s.mean.shape == (4, plant.nu)
    assert mpc.ctrl_steps == 30  # 0.6 / 0.02


def test_optimize_returns_costs_and_keeps_shapes(mpc, plant):
    d0 = plant.make_data()
    s = mpc.init_state()
    s1, costs = jax.jit(mpc.optimize)(d0, 0.0, s, jax.random.PRNGKey(0))
    assert costs.shape == (32,)
    assert jnp.all(jnp.isfinite(costs))
    assert s1.mean.shape == (4, plant.nu)
    assert jnp.all(s1.mean >= plant.u_min) and jnp.all(s1.mean <= plant.u_max)


def test_optimize_reduces_expected_cost_over_iterations(mpc, plant):
    # From a fixed state, repeated MPPI updates at the same time should not
    # increase the cost of the mean rollout (monotone in expectation; check
    # first vs. last with slack).
    d0 = plant.from_state(jnp.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    s = mpc.init_state()
    key = jax.random.PRNGKey(1)
    first = None
    opt = jax.jit(mpc.optimize)
    for _ in range(6):
        key, k = jax.random.split(key)
        s, costs = opt(d0, 0.0, s, k)
        mean_cost = mpc.rollout_cost(d0, s.mean[None])[0]
        first = mean_cost if first is None else first
    assert float(mean_cost) <= float(first) * 1.05


def test_warm_start_shifts_knots_by_time(mpc):
    s = mpc.init_state(initial_knots=jnp.array([[0.0], [1.0], [0.0], [-1.0]]))
    d0 = mpc.plant.make_data()
    s1, _ = mpc.optimize(d0, 0.3, s, jax.random.PRNGKey(0))
    assert jnp.allclose(s1.tk, s.tk + 0.3)


def test_get_action_evaluates_spline_at_t(mpc):
    s = MpcState(tk=jnp.array([0.0, 0.2, 0.4, 0.6]), mean=jnp.array([[0.0], [1.0], [0.0], [-1.0]]))
    assert float(mpc.get_action(s, 0.1)[0]) == pytest.approx(0.5)  # linear midpoint
    assert float(mpc.get_action(s, 0.2)[0]) == pytest.approx(1.0)


def test_step_returns_action_and_state(mpc, plant):
    d0 = plant.make_data()
    u, s1 = jax.jit(mpc.step)(d0, 0.0, mpc.init_state(), jax.random.PRNGKey(0))
    assert u.shape == (plant.nu,)
    assert isinstance(s1, MpcState)


def _randomize_mass(model, key):
    scale = jax.random.uniform(key, (), minval=0.5, maxval=2.0)
    return {"body_mass": model.body_mass * scale}


def _dr_mpc(plant, seed=7):
    return SamplingMpc(
        plant,
        running_cost,
        terminal_cost,
        num_samples=8,
        plan_horizon=0.2,
        noise_level=0.2,
        temperature=0.1,
        num_randomizations=3,
        randomize_model=_randomize_mass,
        seed=seed,
    )


def test_dr_builds_batched_model(plant):
    mpc = _dr_mpc(plant)
    assert mpc.model.body_mass.shape == (3,) + plant.model.body_mass.shape
    # Three distinct scalings.
    assert len({float(m[1]) for m in mpc.model.body_mass}) == 3
    # Non-randomised fields keep their shape.
    assert mpc.model.geom_friction.shape == plant.model.geom_friction.shape


def test_dr_optimize_runs_and_averages(plant):
    mpc = _dr_mpc(plant)
    d0 = plant.make_data()
    s, costs = jax.jit(mpc.optimize)(d0, 0.0, mpc.init_state(), jax.random.PRNGKey(0))
    assert costs.shape == (8,)
    assert jnp.all(jnp.isfinite(costs))


def test_dr_requires_randomize_model(plant):
    with pytest.raises(ValueError, match="randomize_model"):
        SamplingMpc(
            plant,
            running_cost,
            terminal_cost,
            num_samples=4,
            plan_horizon=0.2,
            noise_level=0.1,
            temperature=0.1,
            num_randomizations=2,
        )
