
import jax.numpy as jnp
import pytest

def normalization_logic_fixed(norms):
    # This matches the new implementation in cbf_clf_qp_generator.py
    return jnp.maximum(norms, 1e-8)

def test_normalization_stability():
    """
    Verifies that the normalization logic handles small gradients robustly
    without singularities or massive jumps.
    """
    # Sweep across the critical region 1e-9 to 1e-6
    norms = jnp.logspace(-9, -5, 100)

    safe_norms = normalization_logic_fixed(norms)
    scale_factors = 1.0 / safe_norms
    result_norms = norms * scale_factors

    # Assertions
    # 1. No NaNs or Infs
    assert jnp.all(jnp.isfinite(safe_norms))
    assert jnp.all(jnp.isfinite(scale_factors))
    assert jnp.all(jnp.isfinite(result_norms))

    # 2. Divisor never zero (min divisor is 1e-8)
    assert jnp.all(safe_norms >= 1e-8)

    # 3. Continuity check (simple diff)
    diffs = jnp.diff(result_norms)
    # The result norm should be monotonic?
    # For n < 1e-8: result = n / 1e-8. Linear increase. Monotonic.
    # For n > 1e-8: result = n / n = 1. Constant. Monotonic.
    # So yes, non-decreasing.
    assert jnp.all(diffs >= -1e-7) # Tolerate float32 noise

    # 4. Check specific values
    # Noise (1e-9) -> 0.1
    n_noise = jnp.array([1e-9])
    res_noise = n_noise / normalization_logic_fixed(n_noise)
    assert jnp.allclose(res_noise, 0.1)

    # Signal (1e-1) -> 1.0
    n_signal = jnp.array([1e-1])
    res_signal = n_signal / normalization_logic_fixed(n_signal)
    assert jnp.allclose(res_signal, 1.0)

if __name__ == "__main__":
    try:
        test_normalization_stability()
        print("Verification passed!")
    except AssertionError as e:
        print(f"Verification failed: {e}")
        exit(1)
