
import unittest
from unittest.mock import patch
import jax.numpy as jnp
import jax
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData, EMPTY_CERTIFICATE_COLLECTION

class TestQPSolverSafety(unittest.TestCase):
    def test_max_iter_safety(self):
        """
        Regression test: Verify controller behavior when QP solver hits MAX_ITER_REACHED (status 2).

        Note: The controller logic was updated to treat status 2 as a success (best-effort),
        returning the candidate solution instead of NaNs. This test ensures that:
        1. The controller returns a valid control input (not NaN).
        2. The error flag is False (since status 2 is now considered success).
        3. The internal solver status is correctly reported as 2.
        """

        # 1. Setup simple controller
        control_limits = jnp.array([10.0, 10.0])

        def dynamics(x):
            return jnp.zeros((2,)), jnp.eye(2)

        # We need to construct the controller inside the test or use the pre-generated one.
        # vanilla_cbf_clf_qp_controller is a generator function.
        # It imports solve_qp from cbf_clf_qp_generator module.

        # 2. Define the mock solver
        from collections import namedtuple
        MockState = namedtuple("MockState", ["iter_num"])

        def mock_solve(p_mat, q_vec, g_mat, h_vec, init_params=None):
            # Return dummy solution (zeros), status 2 (MAX_ITER), and None params
            # Solution size must match q_vec (which is n_controls + slacks)
            n_vars = q_vec.shape[0]
            # status=2 means MAX_ITER_REACHED
            # Scout: Controller expects (sol, state) in params to extract iter_num
            mock_state = MockState(iter_num=jnp.array(100))
            return jnp.zeros((n_vars,)), jnp.array(2, dtype=jnp.int32), (None, mock_state)

        # 3. Patch solve_qp in the module where it is used
        with patch('cbfkit.controllers.cbf_clf.cbf_clf_qp_generator.solve_qp', side_effect=mock_solve):

            # Generate the controller *while patched* so it captures the mock?
            # cbf_clf_qp_generator returns a function `generate_cbf_clf_controller`.
            # That function returns `controller`.
            # `controller` uses `solve_qp` from global scope.
            # So patching the global in the module should work even if generator was imported earlier.

            controller = vanilla_cbf_clf_qp_controller(
                control_limits=control_limits,
                dynamics_func=dynamics,
                barriers=EMPTY_CERTIFICATE_COLLECTION,
                lyapunovs=EMPTY_CERTIFICATE_COLLECTION
            )

            # 4. Run the controller
            t = 0.0
            x = jnp.array([0.0, 0.0])
            u_nom = jnp.array([1.0, 1.0])
            key = jax.random.PRNGKey(0)
            data = ControllerData(
                error=False,
                error_data=0,
                complete=False,
                sol=jnp.array([]),
                u=jnp.zeros(2),
                u_nom=jnp.zeros(2),
                sub_data={}
            )

            # JIT compilation happens here
            u, new_data = controller(t, x, u_nom, key, data)

            # 5. Assertions
            # The controller should return valid values (not NaN) for status 2
            self.assertFalse(jnp.isnan(u).any(), f"Controller returned NaNs {u} despite valid status 2!")

            # Error flag should be False (success includes status 2)
            self.assertFalse(new_data.error, "Controller reported error for status 2 (should be success)!")

            # Error data (solver status) should capture the status code 2
            self.assertEqual(new_data.error_data, 2, f"Controller reported wrong status: {new_data.error_data}")

if __name__ == '__main__':
    unittest.main()
