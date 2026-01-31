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

if __name__ == '__main__':
    unittest.main()
