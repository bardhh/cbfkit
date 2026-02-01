import unittest
import jax.numpy as jnp
from jax import random
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

class TestSolverFailureSafety(unittest.TestCase):
    """
    Regression test for QP solver failure handling.

    Ensures that when the QP solver fails (e.g., due to primal infeasibility),
    the controller:
    1. Sets the error flag to True.
    2. Returns NaN values for the control input (fail-loud).
    3. Does NOT return the nominal control input or an invalid "solution".

    This prevents unsafe behavior in downstream systems that might otherwise
    execute a garbage control input.
    """

    def test_infeasible_problem_handling(self):
        # Scenario:
        # State x = -10.
        # Barrier h(x) = x >= 0 implies u >= 10 (roughly, with alpha=1).
        # Control limits |u| <= 1.
        # This is primal infeasible if relaxation is disabled.

        def dynamics(x):
            # x_dot = u
            return jnp.zeros((1,)), jnp.ones((1, 1))

        def h(t, x): return x[0]
        def grad(t, x): return jnp.array([1.0])
        def hess(t, x): return jnp.zeros((1, 1))
        def partial_t(t, x): return 0.0
        def condition(val): return 1.0 * val

        barriers = ([h], [grad], [hess], [partial_t], [condition])

        # Disable relaxation to force hard constraints and induce infeasibility
        ctl = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([1.0]),
            dynamics_func=dynamics,
            barriers=barriers,
            relaxable_cbf=False,
            relaxable_clf=False,
        )

        t = 0.0
        x = jnp.array([-10.0])
        u_nom = jnp.array([0.0])
        key = random.PRNGKey(0)
        data = ControllerData()

        # Execute controller
        u, data = ctl(t, x, u_nom, key, data)

        # 1. Error flag must be set
        self.assertTrue(data.error, f"Controller failed to report error for infeasible problem. Status: {data.error_data}")

        # 2. Status code should NOT be 1 (SOLVED)
        # Note: cbf_clf_qp_generator maps NaN solutions (if any) to status 0.
        # If solver returns -1/-2 (infeasible), it should persist as non-1, or be mapped to 0.
        self.assertNotEqual(data.error_data, 1, "Controller claimed success (status 1) for infeasible problem.")

        # 3. Output must be NaN (Fail-Loud)
        # The 'Aegis' logic in cbf_clf_qp_generator should ensure this.
        self.assertTrue(jnp.isnan(u).all(), f"Controller returned non-NaN value {u} for infeasible problem. Expected NaNs.")

if __name__ == '__main__':
    unittest.main()
