
import pytest
import jax.numpy as jnp
from cbfkit.certificates import certificate_package
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers

def test_certificate_n_mismatch():
    """Verifies that certificate_package raises ValueError when input state dimension
    does not match the specified 'n'."""

    # Define a simple barrier h(x) = x[0] - 1.0
    def cbf_factory():
        def cbf(xt):
            return xt[0] - 1.0
        return cbf

    # Package with n=0 (default) or wrong n
    # Case 1: n=0 (trap)
    pkg_zero = certificate_package(cbf_factory, n=0)
    conditions = zeroing_barriers.linear_class_k(1.0)
    coll_zero = pkg_zero(conditions)
    grad_func_zero = coll_zero.jacobians[0]

    t = 0.0
    x = jnp.array([2.0, 3.0]) # dim 2

    # This should raise ValueError with the fix.
    # Currently it returns empty array (silent failure).
    try:
        g = grad_func_zero(t, x)
        # If we are here, it didn't raise.
        # Check if it returned empty array (current behavior)
        if g.size == 0:
            pytest.fail("Silent failure: n=0 resulted in empty gradient instead of raising ValueError.")
        else:
            # If it returned something else, it's unexpected behavior?
            pass
    except ValueError as e:
        assert "dimension mismatch" in str(e).lower()

    # Case 2: n=1 (wrong dimension, but > 0)
    pkg_wrong = certificate_package(cbf_factory, n=1)
    coll_wrong = pkg_wrong(conditions)
    grad_func_wrong = coll_wrong.jacobians[0]

    try:
        g = grad_func_wrong(t, x)
        # Currently it returns gradient sliced to size 1: [1.0]
        # This is WRONG for 2D system.
        pytest.fail("Silent failure: n=1 resulted in sliced gradient instead of raising ValueError.")
    except ValueError as e:
        assert "dimension mismatch" in str(e).lower()

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_certificate_n_mismatch()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
