import os

# Force JAX to use CPU backend before any jax import occurs.
# This prevents Metal/GPU initialization failures in CI and sandboxed environments.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.random
import pytest


@pytest.fixture
def prng_key():
    """Standard PRNG key for reproducible tests."""
    return jax.random.PRNGKey(0)
