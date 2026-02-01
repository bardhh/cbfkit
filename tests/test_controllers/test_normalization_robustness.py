
import jax.numpy as jnp
import pytest

def normalization_logic_fixed(row_norm):
    # Matches the new implementation in cbf_clf_qp_generator.py
    high, low = 1e-8, 1e-9
    scale_high = 1.0 / high
    slope = (scale_high - 1.0) / (high - low)
    scale_factor = lambda n: 1.0 + (n - low) * slope

    safe_scale = jnp.where(
        row_norm > high,
        1.0 / row_norm,
        jnp.where(row_norm < low, 1.0, scale_factor(row_norm)),
    )
    return row_norm * safe_scale

def test_normalization_stability():
    """
    Verifies that the normalization logic handles small gradients robustly
    without singularities or massive jumps.
    """
    # Sweep across the critical region 1e-10 to 1e-7
    norms = jnp.logspace(-10, -7, 1000)

    result_norms = normalization_logic_fixed(norms)

    # Assertions
    # 1. No NaNs or Infs
    assert jnp.all(jnp.isfinite(result_norms))

    # 2. Monotonicity check
    diffs = jnp.diff(result_norms)
    # The result norm should be monotonic non-decreasing
    # Allow small negative diff due to float32 precision noise (eps ~ 1e-7)
    assert jnp.all(diffs >= -2e-7)

    # 3. Check specific values
    # Noise (1e-10) -> 1e-10 (Unscaled)
    n_low = jnp.array([1e-10])
    res_low = normalization_logic_fixed(n_low)
    assert jnp.allclose(res_low, 1e-10)

    # Signal (1e-7) -> 1.0 (Normalized)
    n_high = jnp.array([1e-7])
    res_high = normalization_logic_fixed(n_high)
    assert jnp.allclose(res_high, 1.0)

    # Transition (5e-9)
    # scale approx 0.5 * 1e8 = 5e7
    # res approx 5e-9 * 5e7 = 0.25
    n_mid = jnp.array([5.5e-9])
    res_mid = normalization_logic_fixed(n_mid)
    # Based on stress test: 5.0e-9 -> 0.22, 5.5e-9 -> ~0.27
    assert res_mid > 0.1
    assert res_mid < 0.5

if __name__ == "__main__":
    try:
        test_normalization_stability()
        print("Verification passed!")
    except AssertionError as e:
        print(f"Verification failed: {e}")
        exit(1)
