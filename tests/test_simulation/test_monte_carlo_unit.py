import unittest
from cbfkit.simulation import monte_carlo
import jax.numpy as jnp

def simple_func(trial_no, **kwargs):
    return trial_no, kwargs.get('key')

def simple_func_no_kwargs(trial_no):
    return trial_no

class TestMonteCarloUnit(unittest.TestCase):
    def test_legacy_behavior(self):
        # Should work without seed and without kwargs support in func if no kwargs passed
        results = monte_carlo.conduct_monte_carlo(simple_func_no_kwargs, n_trials=2)
        self.assertEqual(results, [0, 1])

    def test_seeded_behavior(self):
        # Should inject key if seed provided
        results = monte_carlo.conduct_monte_carlo(simple_func, n_trials=2, seed=10)
        # results is list of (trial_no, key)
        self.assertEqual(results[0][0], 0)
        self.assertEqual(results[1][0], 1)

        # Keys should be present
        self.assertIsNotNone(results[0][1])
        self.assertIsNotNone(results[1][1])

        # Keys should be different
        # (JAX keys are arrays, verify inequality)
        self.assertFalse(jnp.array_equal(results[0][1], results[1][1]))

    def test_seeded_behavior_repeatability(self):
         results1 = monte_carlo.conduct_monte_carlo(simple_func, n_trials=1, seed=10)
         results2 = monte_carlo.conduct_monte_carlo(simple_func, n_trials=1, seed=10)

         # Same seed -> Same key
         self.assertTrue(jnp.array_equal(results1[0][1], results2[0][1]))

    def test_unseeded_behavior_randomness(self):
        # When seed is None, we expect different keys/results across trials
        results = monte_carlo.conduct_monte_carlo(simple_func, n_trials=5, seed=None)
        # results: list of (trial_no, key)

        # Check that keys are not None (if simple_func accepts key)
        # simple_func returns kwargs.get('key')

        keys = [res[1] for res in results]

        # Keys should be present (not None) because simple_func accepts **kwargs
        for k in keys:
            self.assertIsNotNone(k)

        # Keys should be different
        # Compare first two
        self.assertFalse(jnp.array_equal(keys[0], keys[1]))

        # Check all unique
        # Convert to bytes for set hashing
        # Note: keys are JAX arrays
        import numpy as np
        key_bytes = [np.array(k).tobytes() for k in keys]
        self.assertEqual(len(set(key_bytes)), 5)

if __name__ == '__main__':
    unittest.main()
