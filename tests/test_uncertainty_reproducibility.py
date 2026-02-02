
import pytest
import numpy as np
import warnings
from cbfkit.utils.uncertainty import generate_uncertainty_pmf

def test_generate_uncertainty_pmf_warning():
    """Test that a warning is issued when rng is None."""
    u = np.array([1.0, 0.5])
    x = np.array([0.0, 0.0, 1.0, 0.0])
    noise_params = [
        [0.1, 0.1, 0.1, 0.1],
        [0.01, 0.0, 0.0, 0.01]
    ]
    S = 5

    with pytest.warns(UserWarning, match="Using global numpy random state"):
        generate_uncertainty_pmf(u, x, noise_params, S, rng=None)

def test_generate_uncertainty_pmf_reproducibility():
    """Test that providing a seed produces reproducible results."""
    u = np.array([1.0, 0.5])
    x = np.array([0.0, 0.0, 1.0, 0.0])
    noise_params = [
        [0.1, 0.1, 0.1, 0.1],
        [0.01, 0.0, 0.0, 0.01]
    ]
    S = 5
    seed = 42

    # Run 1
    pmf1, u1, x1 = generate_uncertainty_pmf(u, x, noise_params, S, rng=seed)

    # Run 2
    pmf2, u2, x2 = generate_uncertainty_pmf(u, x, noise_params, S, rng=seed)

    np.testing.assert_array_equal(pmf1, pmf2)
    np.testing.assert_array_equal(u1, u2)
    np.testing.assert_array_equal(x1, x2)

def test_generate_uncertainty_pmf_randomness():
    """Test that providing different seeds produces different results."""
    u = np.array([1.0, 0.5])
    x = np.array([0.0, 0.0, 1.0, 0.0])
    noise_params = [
        [0.1, 0.1, 0.1, 0.1],
        [0.01, 0.0, 0.0, 0.01]
    ]
    S = 5

    # Run 1
    _, u1, x1 = generate_uncertainty_pmf(u, x, noise_params, S, rng=42)

    # Run 2
    _, u2, x2 = generate_uncertainty_pmf(u, x, noise_params, S, rng=43)

    assert not np.array_equal(u1, u2)
    assert not np.array_equal(x1, x2)
