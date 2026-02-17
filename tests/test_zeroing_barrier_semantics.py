
import pytest
import jax.numpy as jnp
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k, cubic_class_k, generic_class_k

def test_linear_class_k_semantics():
    """
    Verifies that linear_class_k(alpha) returns a function that computes `alpha * h`.

    The function returns the term alpha(h) that needs to be ADDED to the LHS of the inequality:
    hdot + alpha(h) >= 0
    which is equivalent to:
    hdot >= -alpha(h)

    where alpha(h) = alpha * h.
    """
    alpha = 2.0
    h_val = 3.0

    func = linear_class_k(alpha)
    result = func(h_val)

    # Expected result is alpha * h (positive for positive alpha/h)
    expected = alpha * h_val

    assert jnp.isclose(result, expected), f"Expected {expected}, but got {result}"

    # Explicitly check it's NOT returning the RHS (-alpha * h)
    assert not jnp.isclose(result, -expected), "Function appears to return RHS (-alpha*h) which contradicts code/new spec"

def test_cubic_class_k_semantics():
    """
    Verifies that cubic_class_k(alpha) returns a function that computes `alpha * h^3`.
    """
    alpha = 2.0
    h_val = 2.0

    func = cubic_class_k(alpha)
    result = func(h_val)

    expected = alpha * (h_val ** 3) # 2 * 8 = 16

    assert jnp.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_generic_class_k_semantics():
    """
    Verifies that generic_class_k(alpha_func) wraps the function correctly.
    """
    def my_alpha(h):
        return h + 1.0

    func = generic_class_k(my_alpha)
    h_val = 5.0

    result = func(h_val)
    expected = 6.0

    assert jnp.isclose(result, expected), f"Expected {expected}, but got {result}"
