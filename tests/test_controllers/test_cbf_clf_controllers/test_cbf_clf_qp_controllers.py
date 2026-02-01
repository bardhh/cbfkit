"""
Test Module for cbfkit.controllers.cbf_clf control laws.

=========================

This module contains unit tests for functionalities in 'cbf_clf_controllers'
from 'cbfkit.controllers.cbf_clf'.
"""

import unittest
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.systems.unicycle.models.accel_unicycle.dynamics import accel_unicycle_dynamics
from cbfkit.utils.user_types import ControllerData, EMPTY_CERTIFICATE_COLLECTION

class TestVanillaCBFCLF(unittest.TestCase):
    def test_single_step(self):
        # Setup
        control_limits = jnp.array([1.0, 1.0])
        dynamics = accel_unicycle_dynamics()

        # No barriers/lyapunovs for basic shape test
        controller = vanilla_cbf_clf_qp_controller(
            control_limits=control_limits,
            dynamics_func=dynamics,
            barriers=EMPTY_CERTIFICATE_COLLECTION,
            lyapunovs=EMPTY_CERTIFICATE_COLLECTION
        )

        t = 0.0
        x = jnp.array([0.0, 0.0, 1.0, 0.0]) # v=1
        u_nom = jnp.array([0.5, 0.5])
        key = jnp.array([0, 0], dtype=jnp.uint32)
        data = ControllerData()

        # Run
        u, new_data = controller(t, x, u_nom, key, data)

        # Verify shapes
        self.assertEqual(u.shape, (2,))
        self.assertEqual(u_nom.shape, (2,))

        # Verify values (should match nominal as no constraints)
        self.assertTrue(jnp.allclose(u, u_nom, atol=1e-4))

        # Run with large u_nom to trigger clipping/QP
        u_nom_large = jnp.array([2.0, 2.0])
        u, new_data = controller(t, x, u_nom_large, key, data)

        # Should be clipped/solved to limits
        # Tolerance relaxed due to QP solver precision
        self.assertTrue(jnp.allclose(u, control_limits, atol=1e-3))

    def test_infeasible_qp_fallback(self):
        """Test that the controller returns NaNs when QP is infeasible."""
        from cbfkit.certificates import certificate_package, concatenate_certificates
        from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k

        # 1. Simple Scalar Dynamics: x_dot = u
        def simple_dynamics(x):
            return jnp.zeros((1,)), jnp.ones((1, 1))

        # 2. CBF: h(x) = x >= 0
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0]
            return func

        package = certificate_package(cbf_factory, n=1)
        barriers = concatenate_certificates(
            package(certificate_conditions=linear_class_k(1.0))  # alpha = 1.0
        )

        # 3. Setup Controller with conflicting limits
        # u in [-1, 1]
        control_limits = jnp.array([1.0])

        controller = vanilla_cbf_clf_qp_controller(
            control_limits=control_limits,
            dynamics_func=simple_dynamics,
            barriers=barriers,
            lyapunovs=EMPTY_CERTIFICATE_COLLECTION,
        )

        t = 0.0
        # x = -10. h(x) = -10.
        # dot_h = u.
        # Condition: dot_h + alpha * h >= 0 => u - 10 >= 0 => u >= 10.
        x = jnp.array([-10.0])

        # Nominal control u_nom = 0.
        u_nom = jnp.array([0.0])
        key = jnp.array([0, 0], dtype=jnp.uint32)
        data = ControllerData()

        # 4. Run Controller
        u, new_data = controller(t, x, u_nom, key, data)

        # 5. Assertions
        # Expect QP failure (error=True means failure)
        self.assertTrue(new_data.error)

        # Expect u to be NaN (Aegis safety fallback)
        self.assertTrue(jnp.isnan(u).all())

        # Expect error_data to contain raw status code (0 for Infeasible/Unsolved)
        # Note: jaxopt + OSQP returns 0 for infeasible in current version
        self.assertEqual(new_data.error_data, 0)

        # Try with u_nom outside limits
        u_nom_out = jnp.array([5.0])
        u_out, new_data_out = controller(t, x, u_nom_out, key, data)
        self.assertTrue(new_data_out.error)
        self.assertTrue(jnp.isnan(u_out).all())

if __name__ == '__main__':
    unittest.main()
