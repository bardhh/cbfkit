import jax.numpy as jnp
import pytest

from cbfkit.controllers.mjx_sampling_mpc.spline import get_interp_func, interp_linear, interp_zero


def test_zero_order_hold_takes_previous_knot():
    tq = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    tk = jnp.array([0.0, 0.25, 0.5])
    knots = jnp.array([[[10.0], [20.0], [30.0]]])  # (B=1, K=3, nu=1)
    out = interp_zero(tq, tk, knots)
    assert out.shape == (1, 6, 1)
    assert jnp.allclose(out[0, :, 0], jnp.array([10, 10, 10, 20, 20, 30]))


def test_zero_order_hold_clamps_before_first_knot():
    out = interp_zero(jnp.array([-1.0]), jnp.array([0.0, 1.0]), jnp.array([[[1.0], [2.0]]]))
    assert float(out[0, 0, 0]) == 1.0


def test_linear_interpolates_midpoints_and_batches():
    tq = jnp.array([0.0, 0.5, 1.0])
    tk = jnp.array([0.0, 1.0])
    knots = jnp.array([[[0.0, 10.0], [2.0, 20.0]], [[1.0, 0.0], [1.0, 0.0]]])  # (B=2, K=2, nu=2)
    out = interp_linear(tq, tk, knots)
    assert out.shape == (2, 3, 2)
    assert jnp.allclose(out[0, :, 0], jnp.array([0.0, 1.0, 2.0]))
    assert jnp.allclose(out[0, :, 1], jnp.array([10.0, 15.0, 20.0]))
    assert jnp.allclose(out[1], 1.0 * jnp.array([[1.0, 0.0]] * 3))


def test_get_interp_func_names():
    assert get_interp_func("zero") is interp_zero
    assert get_interp_func("linear") is interp_linear
    with pytest.raises(ValueError, match="cubic"):
        get_interp_func("cubic")
