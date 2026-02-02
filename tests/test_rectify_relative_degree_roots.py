import jax.numpy as jnp
import pytest
from cbfkit.certificates import rectify_relative_degree

def test_scalar_roots_support():
    """
    Verifies that rectify_relative_degree accepts scalar roots (float and 0-d array)
    and correctly broadcasts them to the required number of roots.
    """

    # Dynamics: Double integrator
    # x = [p, v]
    # dp = v
    # dv = u
    def dynamics(x):
        # f(x) = [x[1], 0]
        # g(x) = [0, 1]
        return jnp.array([x[1], 0.0]), jnp.array([[0.0], [1.0]])

    # Barrier: h(x) = p (x[0])
    # Relative degree is 2.
    # compute_function_list returns [h, Lfh, Lf^2h] (length 3).
    # n_roots = 3 - 1 = 2.

    def h(x):
        return x[0]

    state_dim = 2

    # Case 1: Scalar float
    roots_float = -1.0
    cert_float = rectify_relative_degree(
        function=h,
        system_dynamics=dynamics,
        state_dim=state_dim,
        roots=roots_float,
        form="exponential"
    )
    assert callable(cert_float)

    # Case 2: Scalar 0-d array
    roots_0d = jnp.array(-2.0)
    cert_0d = rectify_relative_degree(
        function=h,
        system_dynamics=dynamics,
        state_dim=state_dim,
        roots=roots_0d,
        form="exponential"
    )
    assert callable(cert_0d)

    # Case 3: 1-d array (existing behavior)
    roots_1d = jnp.array([-1.0]) # 1 provided, 2 needed -> broadcasts to [-1, -1]
    cert_1d = rectify_relative_degree(
        function=h,
        system_dynamics=dynamics,
        state_dim=state_dim,
        roots=roots_1d,
        form="exponential"
    )
    assert callable(cert_1d)

if __name__ == "__main__":
    test_scalar_roots_support()
