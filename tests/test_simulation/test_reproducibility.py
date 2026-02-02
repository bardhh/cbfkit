
import re
import logging
import numpy as np
import pytest
from cbfkit.simulation.monte_carlo import conduct_monte_carlo

def mock_experiment(trial_no, **kwargs):
    # This function uses global np.random state, which is seeded by _map_function
    # based on the worker seed derived from the master seed.
    return np.random.random()

def test_monte_carlo_reproducibility(caplog):
    # 1. Run with seed=None and capture log
    with caplog.at_level(logging.WARNING):
        results1 = conduct_monte_carlo(mock_experiment, n_trials=5, n_processes=1, seed=None)

    # 2. Extract seed from log
    # caplog.text contains the log output
    match = re.search(r"Monte Carlo simulation initialized with random seed: (\d+)", caplog.text)
    if not match:
        raise ValueError(f"Could not find seed in logs. Logs were:\n{caplog.text}")

    captured_seed = int(match.group(1))

    # 3. Run again with captured seed
    # Note: When we pass seed explicitly, it won't print the message
    results2 = conduct_monte_carlo(mock_experiment, n_trials=5, n_processes=1, seed=captured_seed)

    # 4. Compare
    # results should be identical
    np.testing.assert_array_equal(results1, results2)

if __name__ == "__main__":
    # This test is designed to be run with pytest
    print("Please run this test with pytest: pytest tests/test_simulation/test_reproducibility.py")
