
import pytest
import jax.numpy as jnp
from jax.scipy.special import erfinv
from cbfkit.certificates.conditions.barrier_conditions.path_integral_barrier import right_hand_side

def test_path_integral_barrier_risk_bound():
    """
    Verifies that right_hand_side(rho, ...) uses the correct inverse error function term
    corresponding to a one-sided risk bound.

    The expected term involves erfinv(1 - 2*rho), which corresponds to the standard normal
    quantile for probability 1-rho (one-sided).

    The previous implementation incorrectly used erfinv(1 - rho).
    """
    rho = 0.05
    gamma = 0.5
    eta = 1.0
    time_period = 1.0

    # Instantiate the function
    # right_hand_side returns a callable that takes 'integral' as input
    func = right_hand_side(rho, gamma, eta, time_period)

    # Evaluate at integral = 0
    integral_val = 0.0
    result = func(integral_val)

    # Expected calculation:
    # 1 - gamma - sqrt(2*T) * eta * erfinv(1 - 2*rho) + integral
    # We use 1 - 2*rho because for a one-sided risk constraint P(h < 0) <= rho,
    # the bound is determined by Phi^{-1}(1-rho) = sqrt(2) * erfinv(1 - 2*rho).

    expected_term = erfinv(1 - 2 * rho)
    expected = 1 - gamma - jnp.sqrt(2 * time_period) * eta * expected_term + integral_val

    # Allow small tolerance for floating point
    assert jnp.isclose(result, expected), \
        f"Expected {expected} (using erfinv(1-2rho)), but got {result}. " \
        f"Value using erfinv(1-rho) would be {1 - gamma - jnp.sqrt(2 * time_period) * eta * erfinv(1 - rho)}"

if __name__ == "__main__":
    test_path_integral_barrier_risk_bound()
