
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.utils.user_types import ControllerData

def test_shape_robustness():
    # Setup
    def mock_cbf(control_limits, *args, **kwargs):
        n_con = len(control_limits)
        return lambda t, x: (jnp.zeros((0, n_con)), jnp.zeros((0,)), {})
    def mock_clf(control_limits, *args, **kwargs):
        n_con = len(control_limits)
        return lambda t, x: (jnp.zeros((0, n_con)), jnp.zeros((0,)), {})

    generator = cbf_clf_qp_generator(mock_cbf, mock_clf)

    # Case 1: 1D system, scalar input
    print("Testing scalar input...")
    control_limits_1d = jnp.array([1.0])
    controller_1d = generator(
        control_limits=control_limits_1d,
        dynamics_func=lambda x: (jnp.zeros((1,)), jnp.ones((1,1))),
    )

    t = 0.0
    x = jnp.array([0.0])
    u_nom_scalar = jnp.array(0.5)
    key = jnp.array([0, 0], dtype=jnp.uint32)
    data = ControllerData()

    u, _ = controller_1d(t, x, u_nom_scalar, key, data)
    print(f"Scalar input result shape: {u.shape}")
    assert u.shape == (1,), f"Expected (1,), got {u.shape}"
    assert u.ndim == 1

    # Case 2: 2D system, (N, 1) input
    print("Testing (N, 1) input...")
    control_limits_2d = jnp.array([1.0, 1.0])
    controller_2d = generator(
        control_limits=control_limits_2d,
        dynamics_func=lambda x: (jnp.zeros((2,)), jnp.eye(2)),
    )

    x2 = jnp.array([0.0, 0.0])
    u_nom_col = jnp.array([[0.5], [0.5]]) # Shape (2, 1)

    u2, _ = controller_2d(t, x2, u_nom_col, key, data)
    print(f"(N, 1) input result shape: {u2.shape}")
    assert u2.shape == (2,), f"Expected (2,), got {u2.shape}"
    assert u2.ndim == 1

    print("Robustness test passed!")

if __name__ == "__main__":
    test_shape_robustness()
