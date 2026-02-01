"""
Test Module for cbfkit.certificates.packager

=========================

This module contains unit tests for the certificate_package function,
ensuring that gradients and partial derivatives are correctly computed
and sliced for time-varying certificates.
"""

import unittest
import jax.numpy as jnp
from cbfkit.certificates import certificate_package

class TestCertificatePackage(unittest.TestCase):
    def test_time_varying_certificate_2d(self):
        """
        Test a time-varying barrier h(x, t) = x[0] - 2*t for a 2D state.

        Verifies:
        - Value: h(x, t)
        - Gradient (x): [1, 0]
        - Partial (t): -2
        """
        # h(x, t) = x[0] - 2*t
        # Input xt is [x0, x1, t]
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0] - 2 * xt[-1]
            return func

        n = 2 # State dim

        pkg_factory = certificate_package(cbf_factory, n=n)

        # Dummy conditions (identity)
        dummy_conditions = lambda val: val

        cert_collection = pkg_factory(certificate_conditions=dummy_conditions)

        v_func = cert_collection.functions[0]
        j_func = cert_collection.jacobians[0]
        t_func = cert_collection.partials[0]

        # Test point: t=1.0, x=[10.0, 0.0]
        t = 1.0
        x = jnp.array([10.0, 0.0])

        # 1. Check Value
        # h = 10 - 2(1) = 8.
        val = v_func(t, x)
        self.assertAlmostEqual(val, 8.0, places=5)

        # 2. Check Gradient (w.r.t x)
        # h = x0 - 2t. grad_x = [1, 0]
        grad = j_func(t, x)
        self.assertEqual(grad.shape, (2,))
        self.assertTrue(jnp.allclose(grad, jnp.array([1.0, 0.0]), atol=1e-5))

        # 3. Check Partial t
        # partial_t = -2
        pt = t_func(t, x)
        # pt should be scalar
        self.assertEqual(pt.shape, ())
        self.assertAlmostEqual(pt, -2.0, places=5)

    def test_time_varying_certificate_scalar(self):
        """
        Test a time-varying barrier h(x, t) = x^2 - t for a 1D state.

        Verifies correct slicing when n=1.
        """
        # h(x, t) = x^2 - t
        # Input xt is [x, t]
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0]**2 - xt[1]
            return func

        n = 1
        pkg_factory = certificate_package(cbf_factory, n=n)
        dummy_conditions = lambda val: val
        cert_collection = pkg_factory(certificate_conditions=dummy_conditions)

        v_func = cert_collection.functions[0]
        j_func = cert_collection.jacobians[0]
        t_func = cert_collection.partials[0]

        t = 2.0
        x = jnp.array([3.0])

        # h = 3^2 - 2 = 7
        self.assertAlmostEqual(v_func(t, x), 7.0, places=5)

        # grad_x = 2x = 6
        grad = j_func(t, x)
        self.assertEqual(grad.shape, (1,))
        self.assertAlmostEqual(grad[0], 6.0, places=5)

        # partial_t = -1
        pt = t_func(t, x)
        self.assertAlmostEqual(pt, -1.0, places=5)

if __name__ == '__main__':
    unittest.main()
