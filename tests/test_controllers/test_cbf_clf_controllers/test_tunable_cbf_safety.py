
import unittest
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)
from cbfkit.utils.user_types import ControllerData, CertificateCollection

class TestTunableCBFSafety(unittest.TestCase):
    def test_tunable_parameter_positivity(self):
        """
        Verifies that tunable Class K parameters (delta) are constrained to be non-negative.
        Negative delta allows inverting the safety condition (h_dot >= -alpha*h becomes h_dot >= alpha*|h|),
        effectively permitting the system to become MORE unsafe when h < 0.
        """
        # System: x is scalar.
        # x_dot = u + drift

        # h(x) = x.
        # Unsafe if x < 0.

        # Common barrier defs
        def h(t, x): return x[0]
        def grad_h(t, x): return jnp.array([1.0])
        def hess_h(t, x): return jnp.zeros((1, 1))
        def partial_t(t, x): return 0.0
        def condition(val): return 1.0 * val # alpha = 1

        barriers = CertificateCollection([h], [grad_h], [hess_h], [partial_t], [condition])

        generator = cbf_clf_qp_generator(
            generate_compute_zeroing_cbf_constraints,
            generate_compute_vanilla_clf_constraints
        )

        t = 0.0
        x = jnp.array([-1.0]) # h(x) = -1 (Unsafe)
        u_nom = jnp.array([0.0])
        key = jnp.array([0, 0], dtype=jnp.uint32)
        data = ControllerData(
            error=False, error_data=0, complete=False,
            sol=jnp.array([]), u=jnp.zeros(1), u_nom=jnp.zeros(1), sub_data={}
        )

        # --- Scenario 1: Recovery Possible ---
        # Lfh = -0.5. Lgh = 1.
        # Constraint: u + (-0.5) + delta * (-1) >= 0 => u >= 0.5 + delta.
        # u in [-1, 1].
        # Feasible for delta > 0?
        # If delta = 0.1 -> u >= 0.6. OK.

        def dynamics_s1(x):
            return jnp.array([-0.5]), jnp.array([[1.0]])

        controller_s1 = generator(
            control_limits=jnp.array([1.0]),
            dynamics_func=dynamics_s1,
            barriers=barriers,
            tunable_class_k=True,
            slack_bound_cbf=100.0,
            relaxable_clf=False
        )

        u_s1, new_data_s1 = controller_s1(t, x, u_nom, key, data)

        self.assertFalse(new_data_s1.error, "Scenario 1 should be feasible")
        sol_s1 = new_data_s1.sol
        delta_s1 = sol_s1[1]

        # Expect reduced but positive delta
        self.assertGreater(delta_s1, 0.0)
        self.assertLess(delta_s1, 1.0) # Nominal is 1.0, must reduce to satisfy u limit

        # --- Scenario 2: Recovery Impossible (without negative delta) ---
        # Lfh = -2.0.
        # Constraint: u - 2 - delta >= 0 => u >= 2 + delta.
        # u_max = 1. => 1 >= 2 + delta => delta <= -1.

        def dynamics_s2(x):
            return jnp.array([-2.0]), jnp.array([[1.0]])

        controller_s2 = generator(
            control_limits=jnp.array([1.0]),
            dynamics_func=dynamics_s2,
            barriers=barriers,
            tunable_class_k=True,
            slack_bound_cbf=100.0,
            relaxable_clf=False
        )

        u_s2, new_data_s2 = controller_s2(t, x, u_nom, key, data)

        # If regression exists (no positivity constraint), this solves with delta ~ -1.
        # We want it to FAIL (return NaNs).

        if not new_data_s2.error:
             delta_s2 = new_data_s2.sol[1]
             print(f"Scenario 2 solved unexpectedly with delta={delta_s2}")

        self.assertTrue(new_data_s2.error, f"Scenario 2 should be infeasible! Got u={u_s2}, status={new_data_s2.error_data}")
        self.assertTrue(jnp.all(jnp.isnan(u_s2)), "Control should be NaN on failure")

if __name__ == '__main__':
    unittest.main()
