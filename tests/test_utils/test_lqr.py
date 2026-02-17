import jax.numpy as jnp

from cbfkit.utils.lqr import compute_lqr_gain


def test_compute_lqr_gain_double_integrator_reference():
    # Continuous-time double integrator:
    # x_dot = [0 1; 0 0] x + [0; 1] u
    # with Q=I, R=1 has known solution K ~= [1, sqrt(3)].
    a_mat = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    b_mat = jnp.array([[0.0], [1.0]])
    q_mat = jnp.eye(2)
    r_mat = jnp.array([[1.0]])

    gain = compute_lqr_gain(a_mat, b_mat, q_mat, r_mat)

    assert gain.shape == (1, 2)
    assert jnp.all(jnp.isfinite(gain))
    expected = jnp.array([[1.0, jnp.sqrt(3.0)]])
    assert jnp.allclose(gain, expected, rtol=1e-4, atol=1e-4)
