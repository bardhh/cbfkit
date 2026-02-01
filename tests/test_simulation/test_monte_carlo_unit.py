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

if __name__ == '__main__':
    unittest.main()
