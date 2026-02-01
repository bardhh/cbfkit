
import pytest
import jax.numpy as jnp
from jax import random, jit
from jaxopt import OSQP

# Import the module where solve_qp is used
from cbfkit.controllers.cbf_clf import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

def test_max_iter_failure_returns_nan():
    """Test that MAX_ITER_REACHED (or other non-SOLVED status) results in NaN output.

    This test verifies that the controller correctly handles cases where the QP solver
    fails to converge (e.g., due to iteration limits), preventing the use of unsafe
    intermediate results.
    """

    # 1. Setup a problem that requires iterations (feasible but non-trivial)
    # Dynamics: x_dot = u.
    # CLF: V = x^2. dot(V) <= -V. => 2xu <= -x^2.
    # x=1. 2u <= -1 => u <= -0.5.
    # u_nom = 10.0 (far from -0.5).
    # This requires moving u significantly.

    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    def V(t, x): return x[0]**2
    def dVdx(t, x): return jnp.array([2*x[0]])
    def d2Vdx2(t, x): return jnp.array([[2.0]])
    def partial_t(t, x): return 0.0
    def condition(val): return -1.0 * val  # dot(V) <= -V

    lyapunovs = ([V], [dVdx], [d2Vdx2], [partial_t], [condition])

    # 2. Define a patched solve_qp function that uses a crippled OSQP solver
    # We define it here to ensure it's a fresh function object, forcing JAX to trace it.
    @jit
    def patched_solve_qp(h_mat, f_vec, g_mat=None, h_vec=None, a_mat=None, b_vec=None, init_params=None):
        # Create a customized solver locally with maxiter=0 to force failure
        qp = OSQP(maxiter=0, tol=1e-9)

        params_obj = (h_mat, 0.5 * f_vec)
        params_eq = None if (a_mat is None or b_vec is None) else (a_mat, b_vec)
        params_ineq = None if (g_mat is None or h_vec is None) else (g_mat, h_vec)

        sol, state = qp.run(
            init_params=init_params,
            params_obj=params_obj,
            params_eq=params_eq,
            params_ineq=params_ineq,
        )
        return sol.primal, state.status, (sol, state)

    # 3. Patch the module's solve_qp
    original_solve_qp = cbf_clf_qp_generator.solve_qp
    cbf_clf_qp_generator.solve_qp = patched_solve_qp

    try:
        # 4. Create controller with strict constraints (no relaxation)
        # The generator will use our patched_solve_qp
        controller = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([10.0]),
            dynamics_func=dynamics,
            lyapunovs=lyapunovs,
            nominal_input=None,
            relaxable_clf=False  # Crucial to force strict constraint solving
        )

        # 5. Run controller
        x = jnp.array([1.0])
        t = 0.0
        key = random.PRNGKey(0)
        data = ControllerData()
        u_nom = jnp.array([10.0])

        u, data = controller(t, x, u_nom, key, data)

        # 6. Assertions
        # Expect failure because maxiter=0 is insufficient
        assert data.error, "Controller should report error on solver failure"
        assert data.error_data != 1, f"Solver status should not be SOLVED (1), got {data.error_data}"

        # Expect NaN return (Safety Invariant)
        assert jnp.all(jnp.isnan(u)), f"Controller should return NaN on solver failure, got {u}"

    finally:
        # Restore original solve_qp
        cbf_clf_qp_generator.solve_qp = original_solve_qp

if __name__ == "__main__":
    test_max_iter_failure_returns_nan()
