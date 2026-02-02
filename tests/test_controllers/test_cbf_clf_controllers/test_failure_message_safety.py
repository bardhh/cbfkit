
import unittest
import jax
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.utils.user_types import ControllerData

class TestFailureMessageSafety(unittest.TestCase):
    """
    Regression test for crash in failure logging.

    Ensures that when the QP solver fails (e.g., infeasible), the error logging mechanism
    (jax.debug.print) does not crash the program due to formatting errors.
    """

    def test_failure_logging_no_crash(self):
        # 1. Setup system with conflicting constraints to force failure
        def dynamics(x):
            return jnp.zeros((1,)), jnp.ones((1, 1))

        # Constraint: u >= 10
        def h1(t, x): return -10.0
        def grad1(t, x): return jnp.array([1.0]) # Lg h = 1. u >= -Lfh - gamma h = 10.
        # But wait, Lfh=0. gamma*h = -10. u >= 10.

        # Constraint: u <= -10 (via control limits? or another CBF?)
        # Let's use control limits u in [-1, 1]

        barriers = ([h1], [grad1], [jnp.zeros((1,1))], [lambda t, x: 0.0], [lambda val: 1.0*val])

        generator = cbf_clf_qp_generator(
            lambda *args, **kwargs: lambda t, x: (jnp.array([[-1.0]]), jnp.array([-10.0]), {"complete": True}), # u >= 10
            lambda *args, **kwargs: lambda t, x: (jnp.zeros((0, 1)), jnp.zeros((0,)), {})
        )

        # Control limits [-1, 1]. Infeasible with u >= 10.
        controller = generator(
            control_limits=jnp.array([1.0]),
            dynamics_func=dynamics,
            barriers=barriers,
            relaxable_cbf=False
        )

        t = 0.0
        x = jnp.zeros(1)
        u_nom = jnp.zeros(1)
        key = jax.random.PRNGKey(0)
        data = ControllerData(
            error=False, error_data=0, complete=False,
            sol=jnp.array([]), u=jnp.zeros(1), u_nom=u_nom,
            sub_data={}
        )

        # This should fail the solver (Primal Infeasible) and trigger _print_failure.
        # It should NOT raise ValueError.
        try:
            u, res_data = controller(t, x, u_nom, key, data)
        except ValueError as e:
            if "Unused keyword arguments" in str(e):
                self.fail(f"Crash in error logging detected: {e}")
            else:
                raise e

        # Also assert that it actually reported failure, to ensure we tested the path
        self.assertNotEqual(res_data.error_data, 1, "Solver should have failed")

if __name__ == '__main__':
    unittest.main()
