import os
import unittest
from unittest import mock
import jax.random as random
from cbfkit.simulation.monte_carlo import conduct_monte_carlo

def trial(trial_no, key=None):
    if key is None:
        return "No key"
    # Return a random number generated from the key
    return float(random.uniform(key))

class TestCBFKitSeed(unittest.TestCase):
    def test_cbfkit_seed_deterministic(self):
        """Test that CBFKIT_SEED enforces determinism."""
        # Set CBFKIT_SEED to a fixed value
        with mock.patch.dict(os.environ, {"CBFKIT_SEED": "12345"}):
            # Run two separate monte carlo simulations
            results1 = conduct_monte_carlo(trial, n_trials=5, n_processes=1)
            results2 = conduct_monte_carlo(trial, n_trials=5, n_processes=1)

            # Verify results are identical
            self.assertEqual(results1, results2)

            # Verify results are NOT "No key"
            self.assertNotEqual(results1[0], "No key")

    def test_cbfkit_seed_unset_nondeterministic(self):
        """Test that unsetting CBFKIT_SEED results in non-determinism (randomness)."""
        # Ensure CBFKIT_SEED is unset
        with mock.patch.dict(os.environ):
            os.environ.pop("CBFKIT_SEED", None)

            # Run two separate monte carlo simulations
            # Note: seed=None is default
            results1 = conduct_monte_carlo(trial, n_trials=5, n_processes=1)
            results2 = conduct_monte_carlo(trial, n_trials=5, n_processes=1)

            # Verify results are different (highly likely)
            # Since we use entropy, they should be different unless entropy is broken
            self.assertNotEqual(results1, results2)

    def test_cbfkit_seed_invalid(self):
        """Test that invalid CBFKIT_SEED falls back to random."""
        with mock.patch.dict(os.environ, {"CBFKIT_SEED": "invalid_int"}):
            results1 = conduct_monte_carlo(trial, n_trials=5, n_processes=1)
            results2 = conduct_monte_carlo(trial, n_trials=5, n_processes=1)

            # Should be different because it falls back to entropy
            self.assertNotEqual(results1, results2)

if __name__ == "__main__":
    unittest.main()
