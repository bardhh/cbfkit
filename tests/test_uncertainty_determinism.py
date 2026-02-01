import numpy as np
import pytest
from cbfkit.utils.uncertainty import generate_uncertainty_pmf

def test_uncertainty_determinism():
    u_nom = np.array([1.0, 0.5])
    x = np.array([0.0, 0.0, 1.0, 0.0])
    noise_params = [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    S = 10

    # Test 1: Implicit randomness (should be different)
    # Note: We rely on the global state changing.
    _, wu1, _ = generate_uncertainty_pmf(u_nom, x, noise_params, S)
    _, wu2, _ = generate_uncertainty_pmf(u_nom, x, noise_params, S)
    # Note: This might flaky-fail if random chance produces same numbers, but extremely unlikely for float arrays
    assert not np.allclose(wu1, wu2), "Implicit randomness should be non-deterministic"

    # Test 2: Explicit seeding (should be identical)
    seed = 42
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    _, wu3, _ = generate_uncertainty_pmf(u_nom, x, noise_params, S, rng=rng1)
    _, wu4, _ = generate_uncertainty_pmf(u_nom, x, noise_params, S, rng=rng2)

    assert np.allclose(wu3, wu4), "Explicit seeding should produce identical results"

def test_integer_seed_api():
    u_nom = np.array([1.0, 0.5])
    x = np.array([0.0, 0.0, 1.0, 0.0])
    noise_params = [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    S = 10
    seed = 12345

    # Calls with same integer seed
    _, wu1, _ = generate_uncertainty_pmf(u_nom, x, noise_params, S, rng=seed)
    _, wu2, _ = generate_uncertainty_pmf(u_nom, x, noise_params, S, rng=seed)

    assert np.allclose(wu1, wu2), "Integer seed should produce identical results"

    # Calls with different integer seeds
    _, wu3, _ = generate_uncertainty_pmf(u_nom, x, noise_params, S, rng=seed+1)

    assert not np.allclose(wu1, wu3), "Different integer seeds should produce different results"


def test_different_seeds():
    u_nom = np.array([1.0, 0.5])
    x = np.array([0.0, 0.0, 1.0, 0.0])
    noise_params = [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    S = 10

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(43)

    _, wu1, _ = generate_uncertainty_pmf(u_nom, x, noise_params, S, rng=rng1)
    _, wu2, _ = generate_uncertainty_pmf(u_nom, x, noise_params, S, rng=rng2)

    assert not np.allclose(wu1, wu2), "Different seeds should produce different results"


from cbfkit.controllers.adaptive_cvar_cbf import adaptive_cvar_cbf_controller
from cbfkit.utils.user_types import ControllerData
import jax.numpy as jnp
from jax import random

class DummyObstacle:
    def __init__(self):
        self.x_curr = np.array([5.0, 5.0, 0.0, 0.0])
        self.velocity_xy = np.array([0.0, 0.0])
        self.radius = 0.5
        self.noise = [[0.01]*4, [0.01]*4]

def test_controller_determinism():
    # Setup
    dynamics_model = {
        "dt": 0.1,
        "A": np.eye(4),
        "B": np.zeros((4, 2)),
        "u_min": -1.0,
        "u_max": 1.0,
        "radius": 0.5
    }
    # Populate B properly for double integrator or similar
    dynamics_model["A"][0, 2] = 1.0
    dynamics_model["A"][1, 3] = 1.0
    dynamics_model["B"][2, 0] = 1.0
    dynamics_model["B"][3, 1] = 1.0

    obstacles = [DummyObstacle()]

    controller = adaptive_cvar_cbf_controller(dynamics_model, obstacles, params={"S": 5})

    t = 0.0
    x = jnp.array([0.0, 0.0, 1.0, 0.0])
    u_nom = jnp.array([1.0, 0.5])
    data = ControllerData()

    key = random.PRNGKey(42)

    # Run 1
    u1, _ = controller(t, x, u_nom, key, data)

    # Run 2 (Same key)
    u2, _ = controller(t, x, u_nom, key, data)

    # Check equality
    assert jnp.allclose(u1, u2), "Controller should be deterministic with same key"

    # Run 3 (Different key)
    key_diff = random.PRNGKey(43)
    u3, _ = controller(t, x, u_nom, key_diff, data)
    # Just verify it runs without error
