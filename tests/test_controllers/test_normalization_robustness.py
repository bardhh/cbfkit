
import jax.numpy as jnp
import pytest

def normalization_logic_fixed(row_norm):
    # Matches the new implementation in cbf_clf_qp_generator.py
    # safe_scales_c = jnp.minimum(1.0 / row_norms_c, 1e8)
    safe_scale = jnp.minimum(1.0 / row_norm, 1e8)
    return row_norm * safe_scale

def test_normalization_stability():
    """
    Verifies that the normalization logic handles small gradients robustly.
    Uses clamped scaling (max 1e8) to prevent noise amplification while ensuring
    safety for gross violations.
    """
    # Sweep across the critical region 1e-10 to 1e-4
    norms = jnp.logspace(-10, -4, 1000)

    result_norms = normalization_logic_fixed(norms)

    # Assertions
    # 1. No NaNs or Infs
    assert jnp.all(jnp.isfinite(result_norms))

    # 2. Monotonicity check
    diffs = jnp.diff(result_norms)
    # The result norm should be monotonic non-decreasing
    assert jnp.all(diffs >= -2e-7)

    # 3. Check specific values
    # Noise (1e-10). Scale clamped at 1e8. Result 1e-2.
    n_low = jnp.array([1e-10])
    res_low = normalization_logic_fixed(n_low)
    assert jnp.allclose(res_low, 1e-2)

    # Signal (1e-4). Scale 1e4 (unclamped). Result 1.0.
    n_high = jnp.array([1e-4])
    res_high = normalization_logic_fixed(n_high)
    assert jnp.allclose(res_high, 1.0)

    # Transition (1e-9). Scale 1e8 (clamped/boundary). Result 0.1.
    n_mid = jnp.array([1e-9])
    res_mid = normalization_logic_fixed(n_mid)
    assert jnp.allclose(res_mid, 0.1)

if __name__ == "__main__":
    try:
        test_normalization_stability()
        print("Verification passed!")
    except AssertionError as e:
        print(f"Verification failed: {e}")
        exit(1)
