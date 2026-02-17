
import pytest
import jax.numpy as jnp
from jax import random
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

def test_sub_atomic_gradient_normalization():
    """
    Test that the controller robustly normalizes extremely small gradients (1e-25).
    This simulates constraints at sub-atomic scales or with poor unit scaling.
    Constraint: 1e-25 * u >= 1e-25  => u >= 1.0
    Cost: u^2
    Expected: u = 1.0
    """
    # 1D system: x_dot = u
    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    # Barrier: h(x)
    # Lfh = 0
    # Lgh = 1e-25
    # alpha(h) = -1e-25 (forcing u >= 1)

    scale = 1e-25

    def h(t, x): return -scale
    def dhdx(t, x): return jnp.array([scale]) # Gradient is 1e-25
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
    # The solver should find u approx 1.0.
    # Without normalization, it finds u approx 0.0 (tolerating 1e-25 violation).
    assert jnp.allclose(u, 1.0, atol=1e-3), f"Expected u=1.0, got {u}. Robust normalization failed for scale {scale}."
