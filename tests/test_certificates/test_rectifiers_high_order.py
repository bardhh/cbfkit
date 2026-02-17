import jax.numpy as jnp
import pytest
from cbfkit.certificates import rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers

def test_rectify_relative_degree_high_order_correctness():
    """
    Test that rectify_relative_degree with form='high-order' correctly:
    1. Returns a certificate (fixes UnboundLocalError).
    2. Returns a certificate with relative degree 1 (fixes logic error).
    """
    # Double Integrator System: x = [p, v], dot_p = v, dot_v = u
    def dynamics(x):
        f = jnp.array([x[1], 0.0])
        g = jnp.array([[0.0], [1.0]])
        return f, g

    # Constraint: p >= 0 => h(x) = x[0]
    # Lg h = 0.
    # Lf h = x[1].
    # High order (alpha=1): h_new = Lf h + h = x[1] + x[0].
    # Lg h_new = 1.
    def h(x):
        return x[0]

    state_dim = 2

    # Pass certificate_conditions directly to avoid TypeError
    cert = rectify_relative_degree(
        function=h,
        system_dynamics=dynamics,
        state_dim=state_dim,
        form="high-order",
        certificate_conditions=zeroing_barriers.linear_class_k(1.0)
    )

    # Test point x = [1.0, 1.0]
    t = 0.0
    x = jnp.array([1.0, 1.0])

    # 1. Check if gradients are computable (verifies cbf_grad fix)
    grad = cert.jacobians[0](t, x)

    # 2. Check if the function is rectified (verifies logic fix)
    # Original h(x) = x[0]. Grad = [1, 0].
    # Rectified h(x) = x[0] + x[1]. Grad = [1, 1].
    # We check the 2nd component of the gradient (corresponding to v, which controls u).

    print(f"Gradient: {grad}")

    # If it uses the original function, grad[1] will be 0.0
    # If it uses the rectified function, grad[1] will be 1.0
    assert not jnp.isclose(grad[1], 0.0), "Gradient along control direction is zero! Function was not rectified."
    assert jnp.isclose(grad[1], 1.0), f"Expected gradient component 1.0, got {grad[1]}"
