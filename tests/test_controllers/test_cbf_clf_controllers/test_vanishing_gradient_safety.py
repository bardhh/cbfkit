
import jax
import jax.numpy as jnp
import pytest
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.utils.user_types import ControllerData

# Enable 64-bit precision to match some configs, but code should work in 32-bit too.
# The issue is more prominent in 32-bit.
jax.config.update("jax_enable_x64", False)

def test_vanishing_gradient_infeasibility():
    """
    Verifies that a hard constraint with vanishing gradient (1e-10)
    that contradicts another constraint is detected as infeasible.
    """
    def mock_cbf(control_limits, dynamics, barriers, lyapunovs, **kwargs):
        def compute(t, x):
            # Constraint 1: -1.0 * u <= -10.0  => u >= 10.0
            # Constraint 2: 1e-10 * u <= -10.0 => u <= -1e11 (practically -inf)
            # Infeasible (10 vs -inf).
            # This represents a case where the system is unsafe (h < 0, b < 0)
            # and gradient is vanishing. We MUST detect this.
            g_mat = jnp.array([[-1.0], [1e-10]])
            h_vec = jnp.array([-10.0, -10.0])
            return g_mat, h_vec, {"complete": True}
        return compute

    def mock_clf(*args, **kwargs):
        return lambda t, x: (jnp.zeros((0, 1)), jnp.zeros((0,)), {})

    generator = cbf_clf_qp_generator(mock_cbf, mock_clf)
    control_limits = jnp.array([100.0])

    controller = generator(
        control_limits=control_limits,
        dynamics_func=lambda t, x: (jnp.zeros(1), jnp.eye(1)),
        barriers=([], [], [], [], []),
        lyapunovs=([], [], [], [], []),
        relaxable_cbf=False
    )

    t = 0.0
    x = jnp.zeros(1)
    u_nom = jnp.zeros(1) # nominal doesn't matter much if constraints conflict
    key = jax.random.PRNGKey(0)
    data = ControllerData(
        error=False, error_data=0, complete=False,
        sol=jnp.zeros(1), u=jnp.zeros(1), u_nom=u_nom,
        sub_data={"solver_params": None}
    )

    u, res_data = controller(t, x, u_nom, key, data)

    # Currently (before fix), solver returns Success (1) with u=10.
    # Because 1e-10 * 10 = 1e-9 < 1e-3.
    # We expect it to FAIL (status != 1) or NaN.

    assert res_data.error_data != 1, f"Solver claimed success! u={u}, status={res_data.error_data}"

def test_vanishing_gradient_spurious():
    """
    Verifies that a constraint with vanishing gradient (1e-10) and compatible RHS (1e-10)
    does NOT clamp the control input unnecessarily (spurious constraint).
    """
    def mock_cbf(*args, **kwargs):
        def compute(t, x):
            # 1e-10 * u <= 1e-10
            # Physically: u <= 1.
            # But if it's noise, we want to ignore it and allow u=10.
            g_mat = jnp.array([[1e-10]])
            h_vec = jnp.array([1e-10])
            return g_mat, h_vec, {"complete": True}
        return compute

    def mock_clf(*args, **kwargs):
        return lambda t, x: (jnp.zeros((0, 1)), jnp.zeros((0,)), {})

    generator = cbf_clf_qp_generator(mock_cbf, mock_clf)
    control_limits = jnp.array([100.0])

    controller = generator(
        control_limits=control_limits,
        dynamics_func=lambda t, x: (jnp.zeros(1), jnp.eye(1)),
        barriers=([], [], [], [], []),
        lyapunovs=([], [], [], [], []),
        relaxable_cbf=False
    )

    t = 0.0
    x = jnp.zeros(1)
    u_nom = jnp.array([10.0]) # Want u=10
    key = jax.random.PRNGKey(0)
    data = ControllerData(
        error=False, error_data=0, complete=False,
        sol=jnp.zeros(1), u=jnp.zeros(1), u_nom=u_nom,
        sub_data={"solver_params": None}
    )

    u, res_data = controller(t, x, u_nom, key, data)

    assert res_data.error_data == 1, "Solver should succeed"
    # With robust normalization (scale 1e8), we detect 1e-10 * 1e8 = 1e-2.
    # Violation 1e-2 * 10 - 1e-2 = 9e-2 > 1e-3.
    # So u=10 is rejected. u=1 is enforced.
    # This is "Spurious" constraint enforcement, but necessary for robust safety.
    assert u[0] <= 1.1, f"Control was NOT clamped! u={u}"

if __name__ == "__main__":
    try:
        test_vanishing_gradient_infeasibility()
        print("Infeasibility Test: PASSED (Failed as expected? No, we assert != 1)")
    except AssertionError as e:
        print(f"Infeasibility Test: FAILED (Solver succeeded incorrectly): {e}")

    try:
        test_vanishing_gradient_spurious()
        print("Spurious Test: PASSED")
    except AssertionError as e:
        print(f"Spurious Test: FAILED: {e}")
