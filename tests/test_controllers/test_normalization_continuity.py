
import jax
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.utils.user_types import ControllerData

# Mock generators
def mock_cbf_gen(control_limits, dyn_func, barriers, lyapunovs, **kwargs):
    def compute(t, x):
        # x[0]: gradient magnitude 'g'
        # x[1]: constraint RHS ratio 'ratio' (h = ratio * g)
        g_val = x[0]
        ratio = x[1]
        h_val = ratio * g_val

        # Constraint: g * u <= h  => u <= ratio

        # We assume 1 control input, 1 CBF
        amat = jnp.array([[g_val]]) # (1, 1)
        bvec = jnp.array([h_val])   # (1,)

        return amat, bvec, {}
    return compute

def mock_clf_gen(control_limits, dyn_func, barriers, lyapunovs, **kwargs):
    def compute(t, x):
        return jnp.zeros((0, 1)), jnp.zeros((0,)), {}
    return compute

def test_normalization_continuity():
    """
    Verifies that the controller output is continuous across the normalization threshold.
    Specifically checks that small gradients (1e-9) and slightly larger gradients (1e-8)
    result in consistent control inputs, indicating no massive normalization jump.
    """
    # Setup
    controller_gen = cbf_clf_qp_generator(mock_cbf_gen, mock_clf_gen)

    control_limits = jnp.array([10.0])
    dynamics_func = lambda x: (jnp.zeros((1,)), jnp.eye(1))
    dummy_barriers = ([lambda t,x: 0.0], [lambda t,x: jnp.zeros(1)], [], [], [])

    # Create controller
    controller = controller_gen(
        control_limits=control_limits,
        dynamics_func=dynamics_func,
        barriers=dummy_barriers,
        relaxable_cbf=False
    )

    u_nom = jnp.array([1.0])
    t = 0.0
    key = jax.random.PRNGKey(0)

    data = ControllerData(
        error=0, error_data=0, complete=False,
        sol=jnp.zeros(1), u=jnp.zeros(1), u_nom=jnp.zeros(1), sub_data={}
    )

    # Case 1: g = 1e-9 (Below old threshold)
    # Ratio = 0.1, so constraint implies u <= 0.1
    x_noise = jnp.array([1e-9, 0.1])
    u_noise, _ = controller(t, x_noise, u_nom, key, data)

    # Case 2: g = 1.01e-8 (Above old threshold)
    x_signal = jnp.array([1.01e-8, 0.1])
    u_signal, _ = controller(t, x_signal, u_nom, key, data)

    # Assert that the difference is small
    # Before fix: u_noise ~ 1.0, u_signal ~ 0.1. Diff ~ 0.9.
    # After fix: u_noise ~ 0.1, u_signal ~ 0.1. Diff ~ 0.0.
    assert jnp.allclose(u_noise, u_signal, atol=0.05), \
        f"Control input jump detected! u_noise={u_noise}, u_signal={u_signal}"
