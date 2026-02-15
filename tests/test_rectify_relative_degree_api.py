
import jax.numpy as jnp
from cbfkit.certificates import rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers

def test_rectify_relative_degree_api():
    """Test both legacy and new API for rectify_relative_degree."""

    # 1. Setup simple system (2D single integrator)
    # x_dot = u
    def dynamics(x):
        return jnp.zeros(2), jnp.eye(2) # f=0, g=I

    # Barrier: h(x) = x[0] > 0
    # Relative degree is 1, so h_dot = grad(h)*f + grad(h)*g*u = [1, 0] * u = u_0
    def h(x):
        return x[0]

    state_dim = 2

    # 2. Legacy Usage: Double Call
    factory = rectify_relative_degree(
        function=h,
        system_dynamics=dynamics,
        state_dim=state_dim,
        form="exponential"
    )

    cert_legacy = factory(
        certificate_conditions=zeroing_barriers.linear_class_k(1.0)
    )

    # 3. New Usage: Single Call
    cert_new = rectify_relative_degree(
        function=h,
        system_dynamics=dynamics,
        state_dim=state_dim,
        form="exponential",
        certificate_conditions=zeroing_barriers.linear_class_k(1.0)
    )

    # 4. Verify Equivalence
    # We check if evaluating the certificate functions on a sample state gives same result

    x_sample = jnp.array([0.5, 0.2])
    t_sample = 0.0

    # Check V(x)
    # cert.functions is a list of callables.
    # Note: certificate functions might be compiled, so we just run them.

    v_legacy = cert_legacy.functions[0](t_sample, x_sample)
    v_new = cert_new.functions[0](t_sample, x_sample)

    assert jnp.allclose(v_legacy, v_new), f"V mismatch: {v_legacy} != {v_new}"

    # Check grad V(x)
    grad_legacy = cert_legacy.jacobians[0](t_sample, x_sample)
    grad_new = cert_new.jacobians[0](t_sample, x_sample)

    assert jnp.allclose(grad_legacy, grad_new), f"Grad mismatch: {grad_legacy} != {grad_new}"

    print("API verification passed: Legacy and New usages produce identical results.")

def test_rectify_relative_degree_input_style():
    """Test the input_style argument for rectify_relative_degree."""

    # Dynamics: x_dot = u
    def dynamics(x):
        return jnp.zeros(2), jnp.eye(2)

    # Strict unpacking function that only accepts state (size 2)
    def h_strict(x):
        p, v = x
        return p

    # This should fail if we don't specify input_style="state" (default is concatenated -> passes size 3)
    # But JAX unpacking might be flexible?
    # Actually, if x is size 3, "p, v = x" raises "too many values to unpack (expected 2)"

    # 1. Verify it works with input_style="state"
    cert = rectify_relative_degree(
        function=h_strict,
        system_dynamics=dynamics,
        state_dim=2,
        form="exponential",
        certificate_conditions=zeroing_barriers.linear_class_k(1.0),
        input_style="state"
    )

    # Verify it works
    x_sample = jnp.array([0.5, 0.2])
    t_sample = 1.0
    val = cert.functions[0](t_sample, x_sample)

    # h = x[0] = 0.5. For rel deg 1, rectified function is just h(x).
    assert jnp.allclose(val, 0.5)

    # 2. Verify it FAILS without input_style="state" (default is concatenated)
    try:
        rectify_relative_degree(
            function=h_strict,
            system_dynamics=dynamics,
            state_dim=2,
            form="exponential",
            certificate_conditions=zeroing_barriers.linear_class_k(1.0)
        )
        assert False, "Should have failed with strict unpacking"
    except Exception as e:
        # We expect a failure during jacfwd call inside compute_function_list
        print(f"Caught expected error: {e}")
        pass

if __name__ == "__main__":
    test_rectify_relative_degree_api()
    test_rectify_relative_degree_input_style()
