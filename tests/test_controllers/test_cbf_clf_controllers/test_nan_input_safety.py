import unittest
import jax.numpy as jnp
from jax import random
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

class TestNanInputSafety(unittest.TestCase):
    """
    Regression test for NaN input handling in QP controllers.

    Ensures that when the controller receives NaNs in the state or other inputs,
    it:
    1. Detects the NaN values before calling the solver (or inspects solver output).
    2. Sets the error flag to True (implied by non-success status).
    3. Sets the status code to -1 (NAN_DETECTED).
    4. Returns NaN values for the control input (fail-loud).

    This prevents numerical instabilities or sensor errors from propagating
    silently as "valid" zero controls.
    """

    def test_nan_state_rejection(self):
        # Scenario: Trivial system
        def dynamics(x):
            return jnp.zeros((1,)), jnp.ones((1, 1))

        # Barrier that depends on x so NaN propagates
        def h(t, x): return x[0]
        def grad(t, x): return jnp.array([1.0])
        def hess(t, x): return jnp.zeros((1, 1))
        def partial_t(t, x): return 0.0
        def condition(val): return 1.0 * val

        barriers = ([h], [grad], [hess], [partial_t], [condition])

        ctl = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([1.0]),
            dynamics_func=dynamics,
            barriers=barriers,
        )

        t = 0.0
        # Inject NaN into state
        x = jnp.array([float('nan')])
        u_nom = jnp.array([0.0])
        key = random.PRNGKey(0)
        data = ControllerData()

        # Execute controller
        u, data = ctl(t, x, u_nom, key, data)

        # 1. Output must be NaN (Fail-Loud)
        self.assertTrue(jnp.isnan(u).all(), f"Controller returned non-NaN value {u} for NaN input. Expected NaNs.")

        # 2. Status code should be -2 (NAN_INPUT_DETECTED)
        # This confirms the specific "Sentinel" logic in cbf_clf_qp_generator is active
        self.assertEqual(data.error_data, -2, f"Controller returned status {data.error_data} instead of -2 for NaN input.")

        # 3. Error flag should be True (since status != 1 and != 2)
        self.assertTrue(data.error, "Controller failed to report error for NaN input.")

if __name__ == '__main__':
    unittest.main()
