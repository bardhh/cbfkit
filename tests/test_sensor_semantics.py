
import jax.numpy as jnp
from jax import random, vmap
from cbfkit.sensors.full_state import unbiased_gaussian_noise
import pytest

def test_sensor_variance_consistency():
    """
    Verifies that unbiased_gaussian_noise produces noise consistent with the specified sigma,
    regardless of the time t.
    """
    key = random.PRNGKey(42)
    sigma = jnp.eye(1) * 1.0  # Variance = 1.0
    x = jnp.array([0.0])

    # Run many trials to estimate variance robustly
    n_trials = 10000
    keys = random.split(key, n_trials)

    # Test at t = 1.0 (should be Variance = 1.0)
    results_t1 = vmap(lambda k: unbiased_gaussian_noise(1.0, x, sigma=sigma, key=k))(keys)
    var_t1 = jnp.var(results_t1)

    # Test at t = 0.0 (should be Variance = 1.0 per docstring, but currently ~0.1 due to implementation)
    results_t0 = vmap(lambda k: unbiased_gaussian_noise(0.0, x, sigma=sigma, key=k))(keys)
    var_t0 = jnp.var(results_t0)

    print(f"Variance at t=1.0: {var_t1:.4f}")
    print(f"Variance at t=0.0: {var_t0:.4f}")

    # Allow some statistical fluctuation, but 0.1 is way off 1.0
    assert abs(var_t1 - 1.0) < 0.1, f"Variance at t=1.0 ({var_t1}) deviated from expected (1.0)"

    # This assertion will fail with the current implementation
    assert abs(var_t0 - 1.0) < 0.1, f"Variance at t=0.0 ({var_t0}) deviated from expected (1.0). Current implementation averages 10 samples at t=0, reducing variance."

if __name__ == "__main__":
    test_sensor_variance_consistency()
