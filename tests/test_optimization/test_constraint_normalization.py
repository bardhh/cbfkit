
import pytest
import jax.numpy as jnp
from jax import random
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

def test_tiny_gradient_normalization():
    """
    Test that the controller can solve a QP with extremely small gradients (1e-6)
    by normalizing the constraints, preventing the solver from treating them as satisfied due to tolerance.

    Constraint: 1e-6 * u >= 1e-6  => u >= 1.0
    Cost: u^2
    Expected: u = 1.0
    """
    # 1D system: x_dot = u
    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    # Barrier: h(x)
    # Lfh = 0
    # Lgh = 1e-6
    # alpha(h) = -1e-6 (forcing u >= 1)

    scale = 1e-6

    # We use a dummy h that is always violated by 'scale'
    # Condition alpha(h) = h -> so we need Lfh + Lgh u >= -alpha(h)
    # 0 + scale * u >= -(-scale) = scale
    # scale * u >= scale => u >= 1

    def h(t, x): return -scale
    def dhdx(t, x): return jnp.array([scale]) # Gradient is tiny!
    def d2hdx2(t, x): return jnp.zeros((1,1))
    def partial_t(t, x): return 0.0
    def condition(val): return 1.0 * val

    barriers = ([h], [dhdx], [d2hdx2], [partial_t], [condition])

    # We use relaxable_cbf=False to force strict satisfaction (or failure if not normalized)
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
    assert jnp.allclose(u, 1.0, atol=1e-3), f"Expected u=1.0, got {u}. Normalization might be broken (solver tolerated 1e-6 as 0)."

def test_large_gradient_normalization():
    """
    Test that large gradients don't cause overflow or numerical instability.
    Constraint: 1e6 * u >= 1e6 => u >= 1.0
    """
    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    scale = 1e6
    def h(t, x): return -scale
    def dhdx(t, x): return jnp.array([scale])
    def d2hdx2(t, x): return jnp.zeros((1,1))
    def partial_t(t, x): return 0.0
    def condition(val): return 1.0 * val

    barriers = ([h], [dhdx], [d2hdx2], [partial_t], [condition])

    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([10.0]),
        dynamics_func=dynamics,
        barriers=barriers,
        relaxable_cbf=False
    )

    x = jnp.array([0.0])
    t = 0.0
    u_nom = jnp.array([0.0])
    key = random.PRNGKey(0)
    data = ControllerData()

    u, data = controller(t, x, u_nom, key, data)

    assert not data.error, f"Controller failed with status {data.error_data}"
    assert jnp.allclose(u, 1.0, atol=1e-3), f"Expected u=1.0, got {u}"

def test_pico_gradient_normalization():
    """
    Test that the controller can solve a QP with extremely small gradients (1e-12)
    by normalizing the constraints.
    Constraint: 1e-12 * u >= 1e-12  => u >= 1.0
    """
    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    scale = 1e-12
    def h(t, x): return -scale
    def dhdx(t, x): return jnp.array([scale])
    def d2hdx2(t, x): return jnp.zeros((1,1))
    def partial_t(t, x): return 0.0
    def condition(val): return 1.0 * val

    barriers = ([h], [dhdx], [d2hdx2], [partial_t], [condition])

    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([10.0]),
        dynamics_func=dynamics,
        barriers=barriers,
        relaxable_cbf=False
    )

    x = jnp.array([0.0])
    t = 0.0
    u_nom = jnp.array([0.0])
    key = random.PRNGKey(0)
    data = ControllerData()

    u, data = controller(t, x, u_nom, key, data)

    assert not data.error, f"Controller failed with status {data.error_data}"
    # Tolerance 1e-3 is acceptable given jaxopt default settings
    assert jnp.allclose(u, 1.0, atol=1e-3), f"Expected u=1.0, got {u}. Normalization might be broken (solver tolerated 1e-12 as 0)."
