
import unittest
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.utils.user_types import ControllerData, CertificateCollection
from cbfkit.controllers.cbf_clf.generate_constraints.zeroing_cbfs import generate_compute_zeroing_cbf_constraints
from cbfkit.controllers.cbf_clf.generate_constraints.vanilla_clfs import generate_compute_vanilla_clf_constraints

class TestSafetyFallback(unittest.TestCase):
    """
    Tests that the controller fails safely (returns NaN) when the QP is infeasible,
    rather than silently applying unsafe nominal control.
    """

    def test_infeasible_fallback_is_nan(self):
        # Setup simple dynamics: x_dot = u
        def simple_dynamics(x):
            return jnp.array([0.0]), jnp.array([[1.0]])

        # Barrier: h(x) = -x. Safe if x <= 0.
        # h_dot = -u
        # Condition: -u + (-x) >= 0 => u <= -x
        def simple_barrier(t, x):
            return jnp.array(-x[0]) # Scalar (shape ())

        def simple_barrier_grad(t, x):
            return jnp.array([-1.0]) # Shape (1,)

        def simple_barrier_hess(t, x):
            return jnp.array([[0.0]])

        def simple_barrier_dt(t, x):
            return jnp.array(0.0)

        def simple_class_k(h):
            return h

        # Setup
        control_limits = jnp.array([1.0]) # u in [-1, 1]

        # Barrier Collection
        barriers = CertificateCollection(
            functions=[simple_barrier],
            jacobians=[simple_barrier_grad],
            hessians=[simple_barrier_hess],
            partials=[simple_barrier_dt],
            conditions=[simple_class_k]
        )

        # Generate Controller
        controller_gen = cbf_clf_qp_generator(
            generate_compute_zeroing_cbf_constraints,
            generate_compute_vanilla_clf_constraints
        )

        controller = controller_gen(
            control_limits=control_limits,
            dynamics_func=simple_dynamics,
            barriers=barriers,
            relaxable_cbf=False # Hard constraint
        )

        # Scenario: x = 2.0 (Unsafe). h = -2.0.
        # Condition: u <= -2.0
        # Limits: u in [-1, 1]
        # Result: Infeasible.

        t = 0.0
        x = jnp.array([2.0])
        u_nom = jnp.array([1.0])
        key = jnp.array([0, 0], dtype=jnp.uint32)
        data = ControllerData()

        # Run
        u, new_data = controller(t, x, u_nom, key, data)

        # Assertions
        self.assertTrue(new_data.error, "Controller should report error on infeasibility")
        self.assertTrue(jnp.isnan(u).all(), "Controller should return NaN on infeasibility to prevent unsafe actuation")

if __name__ == '__main__':
    unittest.main()
