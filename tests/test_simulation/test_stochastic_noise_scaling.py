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
    import jax
    from jax import random

    sigma, dt = 0.1, 0.05
    p = generate_stochastic_perturbation(lambda x: sigma * jnp.eye(2), dt)
    x = jnp.zeros(2)
    u = jnp.zeros(2)
    f, g = _dyn(x)

    def one_step(key):
        val = p(x, u, f, g)(key)
        return integrate_with_cached_dynamics(
            x, u, dt, _dyn, runge_kutta_4, f, g, val, perturbation_is_increment=True
        )[0]

    # vmap(PRNGKey) over arange reproduces PRNGKey(i) for i in range(20000) exactly, so these
    # are the same 20,000 samples a Python loop would draw -- batched into one device dispatch
    # instead of 20,000 round trips, each of which had also forced a float() sync.
    keys = jax.vmap(random.PRNGKey)(jnp.arange(20000))
    incs = jax.jit(jax.vmap(one_step))(keys)
    var = float(np.var(np.asarray(incs)))
    assert abs(var - sigma**2 * dt) < 1e-4, f"got {var}, expected {sigma**2*dt}"


def _brownian_variance_curve(use_jit, n_trials=400, n_steps=100):
    """End-to-end: free Brownian motion through sim.execute. Var(x_t) should be sigma^2 * t.

    Returns the measured variance curve and the Euler-Maruyama prediction, both indexed by
    step, so a caller can check the whole growth law or just the endpoint.

    CAUTION -- the two backends index the returned trajectory differently: the eager path's
    states[0] is the state *after* one step, while the JIT path's states[0] is x0 itself
    (so its curve is shifted one step later and starts at variance 0). ``expected`` below
    follows the eager convention. Only the eager caller compares the full curve; the JIT
    caller compares endpoints only, where the one-step offset is a 1% effect against a 20%
    tolerance. Do not switch the JIT test to a whole-curve check without reconciling this.
    """
    from jax import random

    import cbfkit.simulation.simulator as sim
    from cbfkit.controllers.utils import setup_nominal_controller
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import runge_kutta_4 as integrator
    from cbfkit.sensors import perfect as sensor
    from cbfkit.systems import single_integrator

    sigma, dt = 0.1, 0.05
    dyn = single_integrator.two_dimensional_single_integrator()
    pert = generate_stochastic_perturbation(lambda x: sigma * jnp.eye(2), dt)
    zero_nom = setup_nominal_controller(lambda t, x: jnp.zeros(2))  # u = 0 -> pure diffusion
    trajectories = []
    for i in range(n_trials):
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
        trajectories.append(np.asarray(states))
    # states[k] is the state after k+1 integration steps, so t_k = (k + 1) * dt.
    measured = np.array(trajectories)[:, :, 0].var(axis=0)
    expected = sigma**2 * dt * np.arange(1, n_steps + 1)
    return measured, expected


def test_end_to_end_brownian_variance_non_jit():
    """Non-JIT backend: Var(x_t) follows sigma^2 * t at every step of the rollout.

    This is the eager backend's noise-scaling check, so it cannot be routed through the JIT
    path (the two backends do not currently produce identical trajectories for a given seed
    -- test_rng_consistency pins the planner/controller key streams, not the perturbation's).
    The eager step costs ~10x the JIT step, so the horizon is 10 steps rather than the JIT
    twin's 100. That costs nothing in rigor:

    * The trial count stays at 400, so the variance estimator keeps its original precision
      (relative standard error sqrt(2/399) ~ 7%, leaving the 20% tolerance at ~3 sigma).
      The tolerance is NOT widened.
    * The bug under guard is per-step, not cumulative: were the increment folded into the
      drift as a rate (x + Sigma*sqrt(dt)*w*dt), the variance would shrink by a factor
      dt^2 = 2.5e-3 -- a 400x miss that a single step already exposes.
    * The shortened horizon is repaid by asserting the entire curve instead of the endpoint
      alone, which pins the linear growth law rather than one point on it.
    """
    measured, expected = _brownian_variance_curve(use_jit=False, n_trials=400, n_steps=10)
    rel_err = np.abs(measured - expected) / expected
    assert rel_err.max() < 0.2, f"Var(x_t) departs from sigma^2*t: {measured} vs {expected}"


def test_end_to_end_brownian_variance_jit():
    measured, expected = _brownian_variance_curve(use_jit=True)
    var, exp = float(measured[-1]), float(expected[-1])
    assert abs(var - exp) < 0.2 * exp, f"got {var}, expected ~{exp}"  # ~sigma^2*T=0.05
