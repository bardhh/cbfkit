"""Tests for `cbfkit.sensors.full_state.unbiased_gaussian_noise_factory`.

`unbiased_gaussian_noise` re-derives the Cholesky factor of `sigma` (via a
`lax.cond` over freshly-created branch lambdas) on every call, even though
`sigma` is fixed for the lifetime of a simulation run. The factory closes
over a precomputed Cholesky factor and resolves the noise/no-noise branch
in plain Python at build time, so the returned callable's hot path is just
a matrix-vector product.
"""
import time

import jax
import jax.numpy as jnp
import pytest
from jax import random

from cbfkit.sensors.full_state import unbiased_gaussian_noise, unbiased_gaussian_noise_factory


def test_factory_hot_path_has_no_cholesky_in_jaxpr():
    """The factory-built sensor's traced hot path must not contain jnp.linalg.cholesky."""
    dim = 24
    sigma = 0.05 * jnp.eye(dim)
    x = jnp.zeros(dim)
    key = random.PRNGKey(0)

    # Baseline: the per-call function still traces a cholesky every time.
    jaxpr_original = jax.make_jaxpr(
        lambda t, x, key: unbiased_gaussian_noise(t, x, sigma=sigma, key=key)
    )(0.0, x, key)
    assert "cholesky" in str(jaxpr_original).lower()

    sensor = unbiased_gaussian_noise_factory(sigma)
    jaxpr_factory = jax.make_jaxpr(lambda t, x, key: sensor(t, x, key=key))(0.0, x, key)
    assert "cholesky" not in str(jaxpr_factory).lower()


def test_factory_matches_original_for_same_sigma_and_key():
    """Factory-built sensor must be numerically identical to the per-call sensor."""
    dim = 8
    sigma = 0.2 * jnp.eye(dim) + 0.01
    x = jnp.arange(dim, dtype=jnp.float64)
    key = random.PRNGKey(7)

    sensor = unbiased_gaussian_noise_factory(sigma)

    for t in (0.0, 0.5, 3.0):
        y_original = unbiased_gaussian_noise(t, x, sigma=sigma, key=key)
        y_factory = sensor(t, x, key=key)
        assert jnp.allclose(y_original, y_factory)

    # Passing (and ignoring) an explicit sigma= kwarg at call time -- matching the
    # simulator's `sensor(t, x, sigma=sigma, key=key)` call convention -- must not
    # change the result; the factory always uses the covariance baked in at build time.
    y_with_sigma_kwarg = sensor(0.0, x, sigma=sigma, key=key)
    assert jnp.allclose(y_with_sigma_kwarg, unbiased_gaussian_noise(0.0, x, sigma=sigma, key=key))


def test_factory_zero_covariance_is_passthrough():
    """A zero covariance must resolve to a no-noise (passthrough) branch at build time."""
    dim = 5
    sigma = jnp.zeros((dim, dim))
    x = jnp.arange(dim, dtype=jnp.float64)
    key = random.PRNGKey(1)

    sensor = unbiased_gaussian_noise_factory(sigma)
    y = sensor(0.0, x, key=key)
    assert jnp.allclose(y, x)


def test_factory_variance_consistency():
    """Sanity check: the factory-built sensor reproduces the specified covariance."""
    key = random.PRNGKey(42)
    sigma = jnp.eye(1) * 1.0
    x = jnp.array([0.0])
    sensor = unbiased_gaussian_noise_factory(sigma)

    n_trials = 10000
    keys = random.split(key, n_trials)
    results = jax.vmap(lambda k: sensor(1.0, x, key=k))(keys)
    var = jnp.var(results)
    assert abs(var - 1.0) < 0.1, f"Variance ({var}) deviated from expected (1.0)"


def test_old_signature_still_importable_and_working():
    """Back-compat: `unbiased_gaussian_noise` (the un-hoisted, per-call form) still works."""
    from cbfkit.sensors import unbiased_gaussian_noise as sensor  # re-import path used by examples

    sigma = 0.1 * jnp.eye(3)
    x = jnp.zeros(3)
    y = sensor(0.0, x, sigma=sigma, key=random.PRNGKey(0))
    assert y.shape == x.shape
    assert not jnp.allclose(y, x)  # noise was actually applied


@pytest.mark.parametrize("state_dim", [16, 24])
def test_factory_eager_step_cost_regression_guard(state_dim):
    """Coarse eager-path perf guard (matches the simulate_iter Python-loop call
    pattern used whenever a simulation runs outside `lax.scan`, e.g. STL
    trajectory-cost runs or explicit `use_jit=False`).

    `unbiased_gaussian_noise`'s `lax.cond` wraps freshly-created branch lambdas,
    which defeats JAX's eager dispatch cache and forces a fresh XLA compile on
    every single call. Precomputing the Cholesky factor and resolving the
    branch in Python removes `lax.cond` from the hot path entirely, avoiding
    that per-call recompilation. Threshold is deliberately loose (2x, versus a
    measured >100x on the dev machine) to avoid CI flakiness while still
    catching a real regression.
    """
    sigma = 0.05 * jnp.eye(state_dim)
    x = jnp.zeros(state_dim)
    sensor = unbiased_gaussian_noise_factory(sigma)
    num_steps = 50

    # Warm up both paths once so first-call overhead doesn't dominate either measurement.
    warm_key = random.PRNGKey(0)
    unbiased_gaussian_noise(0.0, x, sigma=sigma, key=warm_key).block_until_ready()
    sensor(0.0, x, key=warm_key).block_until_ready()

    keys = random.split(random.PRNGKey(1), num_steps)

    start = time.perf_counter()
    out = x
    for k in keys:
        out = unbiased_gaussian_noise(0.0, out, sigma=sigma, key=k)
    out.block_until_ready()
    t_original = time.perf_counter() - start

    start = time.perf_counter()
    out = x
    for k in keys:
        out = sensor(0.0, out, key=k)
    out.block_until_ready()
    t_factory = time.perf_counter() - start

    print(
        f"[state_dim={state_dim}] eager {num_steps}-step cost: "
        f"original={t_original * 1000:.2f}ms factory={t_factory * 1000:.2f}ms "
        f"speedup={t_original / t_factory:.1f}x"
    )

    assert t_factory < 0.5 * t_original, (
        f"expected the hoisted sensor to be meaningfully faster in the eager path: "
        f"original={t_original * 1000:.2f}ms factory={t_factory * 1000:.2f}ms"
    )
