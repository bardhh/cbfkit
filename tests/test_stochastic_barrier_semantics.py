
import pytest
import jax.numpy as jnp
from cbfkit.certificates.conditions.barrier_conditions.stochastic_barrier import right_hand_side

def test_stochastic_barrier_rhs_semantics():
    """
    Verifies that right_hand_side(alpha, beta) returns a function that computes `alpha * h - beta`.

    The function returns the term that needs to be ADDED to the LHS of the inequality:
    hdot + (alpha * h - beta) >= 0
    which is equivalent to:
    hdot >= -alpha * h + beta.

    This test ensures the implementation remains consistent with this semantic interpretation.
    """
    alpha = 2.0
    beta = 0.5
    h_val = 10.0

    # Instantiate the function
    func = right_hand_side(alpha, beta)

    # Evaluate
    result = func(h_val)

    # The implementation returns alpha * h - beta
    expected = alpha * h_val - beta

    assert jnp.isclose(result, expected), \
        f"Expected {expected} (alpha*h - beta), but got {result}"
