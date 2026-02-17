
import pytest
import jax.numpy as jnp
from jax import random
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

def test_stiff_constraint_safety():
    """
    Test that the controller robustly handles stiff constraints (large coefficients).
    Constraint: 1e5 * u >= 1e5  => u >= 1.0
    With loose tolerance (1e-3 relative), solver might return u=0.999.
    Physical violation: 1e5 * (1 - 0.999) = 100.
    This test ensures physical violation is small (< 0.1).
    """
    # 1D system: x_dot = u
    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    # Barrier: h(x)
    # Lfh = 0
    # Lgh = 1e5
    # alpha(h) = -1e5 (forcing u >= 1)

    scale = 1e5

    def h(t, x): return -scale
    def dhdx(t, x): return jnp.array([scale]) # Gradient is 1e5
    def d2hdx2(t, x): return jnp.zeros((1,1))
    def partial_t(t, x): return 0.0
    def condition(val): return 1.0 * val

    barriers = ([h], [dhdx], [d2hdx2], [partial_t], [condition])

    # We use relaxable_cbf=False to force strict satisfaction
    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([10.0]), # Large enough
        dynamics_func=dynamics,
        barriers=barriers,
        relaxable_cbf=False
    )

    x = jnp.array([0.0])
    t = 0.0
    u_nom = jnp.array([0.0])
    key = random.PRNGKey(0)
    data = ControllerData()

    # Run controller
    u, data = controller(t, x, u_nom, key, data)

    assert not data.error, f"Controller failed with status {data.error_data}"

    # Calculate physical violation
    # Constraint: 1e5 * u >= 1e5
    lhs = scale * u[0]
    rhs = scale
    violation = rhs - lhs

    # Assert physical violation is small (e.g. < 0.1)
    # With tol=1e-3, violation was ~100.
    # With tol=1e-6, violation should be ~0.1 or less.
    assert violation < 0.1, f"Physical constraint violation too large: {violation} (u={u})"
