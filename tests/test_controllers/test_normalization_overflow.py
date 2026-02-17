
import jax
import jax.numpy as jnp
import pytest
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)
from cbfkit.utils.user_types import CertificateCollection, ControllerData

# Enable x64 (standard for cbfkit)
jax.config.update("jax_enable_x64", True)

def test_normalization_overflow():
    """
    Verifies that large gradients (which fit within x64 precision)
    do not cause constraints to vanish.

    Note: cbfkit assumes inputs fit within physically reasonable bounds (< 1e150).
    Here we test 1e100, which is enormous but safe for x64 norm (squares to 1e200).

    Case: Lg h = 1e100, Lf h = -1e100.
    Constraint: Lf h + Lg h * u >= 0  =>  -1e100 + 1e100 * u >= 0  => u >= 1.
    Nominal u = 0.

    Expected: u = 1.
    Failure mode (vanished constraint): u = 0.
    """

    # 1. Define Dynamics
    scale = 1.0e100

    def dynamics(x):
        f = jnp.array([-scale, 0.0])
        g = jnp.array([[scale, 0.0], [0.0, 1.0]])
        return f, g

    def h(t, x): return x[0]
    def grad_h(t, x): return jnp.array([1.0, 0.0])
    def hess_h(t, x): return jnp.zeros((2, 2))
    def partial_t(t, x): return 0.0
    def class_k(val): return 0.0

    barriers = ([h], [grad_h], [hess_h], [partial_t], [class_k])

    # 3. Create Controller
    control_limits = jnp.array([10.0, 10.0])

    gen_controller = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints
    )

    controller = gen_controller(
        control_limits=control_limits,
        dynamics_func=dynamics,
        barriers=barriers,
        lyapunovs=None,
        relaxable_cbf=False,
        relaxable_clf=False,
    )

    # 4. Run Controller
    t = 0.0
    x = jnp.array([0.0, 0.0])
    u_nom = jnp.array([0.0, 0.0])
    key = None
    data = ControllerData(
        error=jnp.array(False),
        error_data=jnp.array(0),
        complete=jnp.array(False),
        sol=jnp.zeros((2,)),
        u=jnp.zeros((2,)),
        u_nom=u_nom,
        sub_data={},
    )

    # Run twice to ensure JIT compilation happens
    # First run might be slow
    print("Running controller (1st pass)...")
    u_actual, data_out = controller(t, x, u_nom, key, data)

    print(f"Computed Control: {u_actual}")
    print(f"Solver Status: {data_out.sub_data.get('solver_status')}")

    # Check Result
    # Expect u[0] >= 1.0 (approx 1.0)
    # If constraint vanished, u[0] == 0.0

    if jnp.any(jnp.isnan(u_actual)):
        pytest.fail(f"Computed control contains NaNs: {u_actual}")

    assert u_actual[0] > 0.9, f"Constraint failed! Expected u[0] >= 1.0, got {u_actual[0]}"
    assert u_actual[0] < 1.1, f"Control overly aggressive? Got {u_actual[0]}"

if __name__ == "__main__":
    try:
        test_normalization_overflow()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"Exception: {e}")
        exit(1)
