import jax.numpy as jnp
from cbfkit.certificates import rectify_relative_degree

def test_rectify_relative_degree_tolerance():
    """
    Verifies that the relative degree determination logic correctly ignores
    vanishing gradients (below 1e-6) which are often numerical artifacts
    or negligible couplings in stiff systems.

    System:
    x = [x0, x1]
    dx0 = x1 + epsilon * u
    dx1 = u

    h(x) = x0

    L_g h = epsilon.

    If epsilon < 1e-6, the rectifier should ignore it and proceed to the next derivative.
    The effective relative degree should be 2.

    If it stops early, it thinks RD=1.
    """

    epsilon = 1e-8

    def dynamics(x):
        # f(x) = [x1, 0]
        # g(x) = [epsilon, 1]
        f = jnp.array([x[1], 0.0])
        g = jnp.array([[epsilon], [1.0]])
        return f, g

    def h(x):
        return x[0]

    # We expect RD=2, so we need 1 root (pole) for the stable polynomial.
    # Let's use roots=[-1.0].
    # Polynomial for RD=2 (s+1) -> s + 1.
    # Coefficients: [1, 1].
    # CBF = 1 * h(x) + 1 * h_dot(x).

    # If RD=1 (failure case), no roots needed (or ignored).
    # CBF = 1 * h(x).

    # Instantiate with dummy conditions to get the CertificateCollection
    def dummy_cond(alpha):
        return lambda x: alpha * x

    cert = rectify_relative_degree(
        function=h,
        system_dynamics=dynamics,
        state_dim=2,
        roots=jnp.array([-1.0]), # Implies we expect RD=2
        form="exponential",
        certificate_conditions=dummy_cond(1.0)
    )

    # State where x0=0, x1=1
    x_test = jnp.array([0.0, 1.0])
    t_test = 0.0

    # Evaluate CBF
    # cert.functions[0] is the V(x)
    val = cert.functions[0](t_test, x_test)

    # Expected:
    # If RD=2 (Correct):
    # CBF = c0*h + c1*h_dot
    # h(x) = x[0] = 0
    # h_dot(x) = L_f h = x[1] = 1
    # CBF = 1*0 + 1*1 = 1.0

    # If RD=1 (Incorrect/Regression):
    # CBF = 1*h = x[0] = 0.0

    assert jnp.isclose(val, 1.0), f"Expected CBF value 1.0 (RD=2), got {val} (RD=1? Regression in tolerance check)"
