import jax.numpy as jnp
from jax import random

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.optimization.quadratic_program.solver_registry import get_solver
from cbfkit.utils.user_types import ControllerData


def test_max_iter_failure_returns_nan():
    """Test that MAX_ITER_REACHED (or other non-SOLVED status) results in NaN output.

    This test verifies that the controller correctly handles cases where the QP solver
    fails to converge (e.g., due to iteration limits), preventing the use of unsafe
    intermediate results.
    """

    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    def V(t, x):
        return x[0] ** 2

    def dVdx(t, x):
        return jnp.array([2 * x[0]])

    def d2Vdx2(t, x):
        return jnp.array([[2.0]])

    def partial_t(t, x):
        return 0.0

    def condition(val):
        return -1.0 * val  # dot(V) <= -V

    lyapunovs = ([V], [dVdx], [d2Vdx2], [partial_t], [condition])

    # Use a crippled solver (maxiter=0) to force failure
    crippled_solver = get_solver("jaxopt", max_iter=0, tol=1e-9)

    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([10.0]),
        dynamics_func=dynamics,
        lyapunovs=lyapunovs,
        relaxable_clf=False,
        solver=crippled_solver,
    )

    x = jnp.array([1.0])
    t = 0.0
    key = random.PRNGKey(0)
    data = ControllerData()
    u_nom = jnp.array([10.0])

    u, data = controller(t, x, u_nom, key, data)

    # Expect failure because maxiter=0 is insufficient
    assert data.error, "Controller should report error on solver failure"
    assert data.error_data != 1, f"Solver status should not be SOLVED (1), got {data.error_data}"

    # Expect NaN return (Safety Invariant)
    assert jnp.all(jnp.isnan(u)), f"Controller should return NaN on solver failure, got {u}"


if __name__ == "__main__":
    test_max_iter_failure_returns_nan()
