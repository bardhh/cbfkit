"""Euler-Maruyama: a stochastic perturbation is an increment, not a dt-scaled rate."""
import jax.numpy as jnp
import numpy as np

from cbfkit.integration import runge_kutta_4
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
from cbfkit.simulation.integration_utils import integrate_with_cached_dynamics


def _dyn(s):
    return jnp.zeros(2), jnp.eye(2)


def test_increment_added_once_not_scaled_by_dt():
    x = jnp.zeros(2)
    u = jnp.zeros(2)
    f, g = _dyn(x)
    dt = 0.05
    incr = jnp.array([0.3, -0.2])
    out_inc = integrate_with_cached_dynamics(
        x, u, dt, _dyn, runge_kutta_4, f, g, incr, perturbation_is_increment=True
    )
    assert np.allclose(np.asarray(out_inc), np.asarray(incr), atol=1e-9)  # x + incr
    out_rate = integrate_with_cached_dynamics(
        x, u, dt, _dyn, runge_kutta_4, f, g, incr, perturbation_is_increment=False
    )
    assert np.allclose(np.asarray(out_rate), np.asarray(incr) * dt, atol=1e-9)  # legacy x + incr*dt


def test_stochastic_perturbation_flagged_as_increment():
    p = generate_stochastic_perturbation(lambda x: 0.1 * jnp.eye(2), 0.05)
    assert getattr(p, "is_increment", False) is True


def test_simulated_variance_is_euler_maruyama():
    # single integrator, no control, sigma=0.1: per-step state increment var ~ sigma^2*dt
    from jax import random

    sigma, dt = 0.1, 0.05
    p = generate_stochastic_perturbation(lambda x: sigma * jnp.eye(2), dt)
    x = jnp.zeros(2)
    f, g = _dyn(x)
    incs = []
    for i in range(20000):
        val = p(x, jnp.zeros(2), f, g)(random.PRNGKey(i))
        nx = integrate_with_cached_dynamics(
            x, jnp.zeros(2), dt, _dyn, runge_kutta_4, f, g, val, perturbation_is_increment=True
        )
        incs.append(float(nx[0]))
    var = float(np.var(np.array(incs)))
    assert abs(var - sigma**2 * dt) < 1e-4, f"got {var}, expected {sigma**2*dt}"


def _final_state_variance(use_jit):
    """End-to-end: free Brownian motion through sim.execute. Var(x_T) should be sigma^2 * T."""
    from jax import random

    import cbfkit.simulation.simulator as sim
    from cbfkit.controllers.utils import setup_nominal_controller
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import runge_kutta_4 as integrator
    from cbfkit.sensors import perfect as sensor
    from cbfkit.systems import single_integrator

    sigma, dt, n_steps = 0.1, 0.05, 100  # T = 5.0
    dyn = single_integrator.two_dimensional_single_integrator()
    pert = generate_stochastic_perturbation(lambda x: sigma * jnp.eye(2), dt)
    zero_nom = setup_nominal_controller(lambda t, x: jnp.zeros(2))  # u = 0 -> pure diffusion
    finals = []
    for i in range(400):
        states, *_ = sim.execute(
            x0=jnp.zeros(2),
            dynamics=dyn,
            sensor=sensor,
            controller=None,
            nominal_controller=zero_nom,
            estimator=estimator,
            integrator=integrator,
            perturbation=pert,
            dt=dt,
            num_steps=n_steps,
            key=random.PRNGKey(i),
            verbose=False,
            use_jit=use_jit,
        )
        finals.append(np.asarray(states[-1]))
    return float(np.var(np.array(finals)[:, 0])), sigma**2 * (dt * n_steps)


def test_end_to_end_brownian_variance_non_jit():
    var, expected = _final_state_variance(use_jit=False)
    assert (
        abs(var - expected) < 0.2 * expected
    ), f"got {var}, expected ~{expected}"  # ~sigma^2*T=0.05


def test_end_to_end_brownian_variance_jit():
    var, expected = _final_state_variance(use_jit=True)
    assert abs(var - expected) < 0.2 * expected, f"got {var}, expected ~{expected}"
