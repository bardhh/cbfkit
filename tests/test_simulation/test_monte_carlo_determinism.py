
import unittest
import numpy as np
from cbfkit.simulation import monte_carlo

def dummy_task(trial_no, **kwargs):
    return trial_no

class TestMonteCarloDeterminism(unittest.TestCase):
    def test_global_random_state_preservation(self):
        """Test that conduct_monte_carlo does not modify the global numpy random state."""

        # Set a known state
        np.random.seed(42)
        state_before = np.random.get_state()
        val1 = np.random.rand()

        # We need to reset to state_before or just check the sequence continues as expected
        # Let's check that the sequence matches what we expect if MC wasn't run.

        np.random.set_state(state_before)
        # Advance by 1
        _ = np.random.rand()
        val2_expected = np.random.rand()

        # Now do the actual run
        np.random.set_state(state_before)
        # Advance by 1 (the val1 above)
        _ = np.random.rand()

        # Run MC
        # We use n_processes=1 to ensure it runs in the current process
        monte_carlo.conduct_monte_carlo(dummy_task, n_trials=5, n_processes=1, seed=123)

        # Check next value
        val2_actual = np.random.rand()

        self.assertEqual(val2_actual, val2_expected, "Global random state was modified by conduct_monte_carlo")

if __name__ == '__main__':
    unittest.main()
