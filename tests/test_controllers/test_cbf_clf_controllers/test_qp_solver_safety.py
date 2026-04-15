import unittest
from collections import namedtuple

import jax
import jax.numpy as jnp

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.optimization.quadratic_program.solver_registry import QpSolution
from cbfkit.utils.user_types import ControllerData, EMPTY_CERTIFICATE_COLLECTION


MockState = namedtuple("MockState", ["iter_num"])


def _mock_max_iter_solver(
    h_mat, f_vec, g_mat=None, h_vec=None, a_mat=None, b_vec=None, init_params=None
):
    """Mock solver that always returns MAX_ITER_REACHED (status 2)."""
    n_vars = f_vec.shape[0]
    mock_state = MockState(iter_num=jnp.array(100))
    return QpSolution(
        primal=jnp.zeros(n_vars),
        status=jnp.array(2, dtype=jnp.int32),
        params=(None, mock_state),
    )


_mock_max_iter_solver.jit_compatible = True
_mock_max_iter_solver.solver_name = "mock"


class TestQPSolverSafety(unittest.TestCase):
    def test_max_iter_safety(self):
        """
        Regression test: Verify handling of MAX_ITER_REACHED (status 2).
        The controller treats status 2 as failure (unsafe to use unconverged result),
        returning NaNs.
        """
        control_limits = jnp.array([10.0, 10.0])

        def dynamics(x):
            return jnp.zeros((2,)), jnp.eye(2)

        # Pass the mock solver via the solver= parameter
        controller = vanilla_cbf_clf_qp_controller(
            control_limits=control_limits,
            dynamics_func=dynamics,
            barriers=EMPTY_CERTIFICATE_COLLECTION,
            lyapunovs=EMPTY_CERTIFICATE_COLLECTION,
            solver=_mock_max_iter_solver,
        )

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
            sub_data={},
        )

        u, new_data = controller(t, x, u_nom, key, data)

        # Controller treats status 2 as failure — u should be NaNs
        self.assertTrue(
            jnp.isnan(u).any(),
            f"Controller returned valid output {u} for MAX_ITER status (should be NaNs)!",
        )
        self.assertTrue(new_data.error, "Controller failed to report error for status 2!")
        self.assertEqual(
            new_data.error_data, 2, f"Controller reported wrong status: {new_data.error_data}"
        )


if __name__ == "__main__":
    unittest.main()
