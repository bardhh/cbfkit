
import unittest
import jax.numpy as jnp
from jax import random
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

class TestHighPenaltyRobustness(unittest.TestCase):
    """
    Regression test for handling high slack variable penalties in CBF-QP.
    Previous versions failed with UNSOLVED/NaN for penalties >= 100.0 due to
    solver iteration limits or ill-conditioning.
    """

    def test_high_penalty_feasibility(self):
        # Scenario: Unsafe state requiring slack usage
        # State: x = -2.0. Barrier h(x) = x >= 0.
        # Condition: u + delta >= 2.0.
        # Limit: u <= 0.5.
        # Must use slack delta >= 1.5.

        def dynamics(x):
            return jnp.array([0.0]), jnp.array([[1.0]])

        def h(t, x): return x[0]
        def grad(t, x): return jnp.array([1.0])
        def hess(t, x): return jnp.array([[0.0]])
        def partial_t(t, x): return 0.0
        def condition(val): return 1.0 * val

        barriers = ([h], [grad], [hess], [partial_t], [condition])
        x = jnp.array([-2.0])
        u_nom = jnp.array([0.0])
        key = random.PRNGKey(0)

        # Penalty 1e5 was definitely failing in benchmarks
        penalty = 1e5

        ctl = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([0.5]),
            dynamics_func=dynamics,
            barriers=barriers,
            nominal_input=None,
            relaxable_cbf=True,
            slack_penalty_cbf=penalty,
        )

        data = ControllerData()
        u, data = ctl(0.0, x, u_nom, key, data)

        # Assertions
        self.assertFalse(data.error, f"Controller reported error for penalty {penalty}. Status: {data.error_data}")
        self.assertFalse(jnp.any(jnp.isnan(u)), f"Controller returned NaN for penalty {penalty}: {u}")

        # Check that slack was used (u should be near limit 0.5)
        # u should be close to 0.5 (clamped)
        # and slack should satisfy constraint.
        self.assertTrue(u[0] <= 0.5 + 1e-3, f"Control input violated limit: {u[0]}")
        # Ideally u is exactly 0.5 because minimizing slack squared means maximizing u to reduce delta.
        self.assertTrue(u[0] >= 0.4, f"Control input suspiciously low: {u[0]}")

if __name__ == '__main__':
    unittest.main()
