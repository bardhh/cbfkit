
import unittest
import jax.numpy as jnp
from cbfkit.certificates import certificate_package

class TestCertificatePackageMismatch(unittest.TestCase):
    def test_n_mismatch_default(self):
        """
        Test that calling a packaged certificate with n=0 (default) but providing a valid state vector
        raises a ValueError.
        """
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0]
            return func

        # n defaults to 0
        pkg_factory = certificate_package(cbf_factory)
        dummy_conditions = lambda val: val
        cert_collection = pkg_factory(certificate_conditions=dummy_conditions)

        j_func = cert_collection.jacobians[0]

        t = 0.0
        x = jnp.array([1.0, 2.0]) # Shape (2,)

        with self.assertRaises(ValueError) as cm:
            j_func(t, x)

        self.assertIn("State dimension mismatch", str(cm.exception))
        self.assertIn("expected input shape (0,)", str(cm.exception))

    def test_n_mismatch_explicit(self):
        """
        Test that calling a packaged certificate with n=1 but providing a state vector of shape (2,)
        raises a ValueError.
        """
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0]
            return func

        # n explicitly set to 1
        pkg_factory = certificate_package(cbf_factory, n=1)
        dummy_conditions = lambda val: val
        cert_collection = pkg_factory(certificate_conditions=dummy_conditions)

        j_func = cert_collection.jacobians[0]

        t = 0.0
        x = jnp.array([1.0, 2.0]) # Shape (2,)

        with self.assertRaises(ValueError) as cm:
            j_func(t, x)

        self.assertIn("State dimension mismatch", str(cm.exception))
        self.assertIn("expected input shape (1,)", str(cm.exception))

    def test_n_match(self):
        """
        Test that calling with correct n proceeds without error.
        """
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0] - 1.0
            return func

        n = 2
        pkg_factory = certificate_package(cbf_factory, n=n)
        dummy_conditions = lambda val: val
        cert_collection = pkg_factory(certificate_conditions=dummy_conditions)

        j_func = cert_collection.jacobians[0]

        t = 0.0
        x = jnp.array([2.0, 3.0]) # Shape (2,)

        # Should not raise
        grad = j_func(t, x)
        self.assertEqual(grad.shape, (2,))

    def test_scalar_n1(self):
        """
        Test that scalar input (0-D array) is accepted if n=1.
        """
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0] - 1.0
            return func

        n = 1
        pkg_factory = certificate_package(cbf_factory, n=n)
        dummy_conditions = lambda val: val
        cert_collection = pkg_factory(certificate_conditions=dummy_conditions)

        j_func = cert_collection.jacobians[0]

        t = 0.0
        x = jnp.array(2.0) # Scalar

        # Should not raise
        grad = j_func(t, x)
        # Gradient of scalar is scalar or 1D?
        # certificate_package usually returns gradients matching state shape?
        # j_func uses jnp.hstack([x, t]). x (scalar) + t (scalar) -> (2,)
        # j_func (auto-diff) returns gradient w.r.t xt.
        # j_func wrapper returns [:n]. n=1. So returns array of size 1.
        self.assertEqual(grad.shape, (1,))

    def test_scalar_n2_fails(self):
        """
        Test that scalar input (0-D array) fails if n=2.
        """
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0] - 1.0
            return func

        n = 2
        pkg_factory = certificate_package(cbf_factory, n=n)
        dummy_conditions = lambda val: val
        cert_collection = pkg_factory(certificate_conditions=dummy_conditions)

        j_func = cert_collection.jacobians[0]

        t = 0.0
        x = jnp.array(2.0) # Scalar

        with self.assertRaises(ValueError) as cm:
            j_func(t, x)

        self.assertIn("State dimension mismatch", str(cm.exception))
        self.assertIn("scalar input (0-D) provided but certificate_package expected n=2", str(cm.exception))

    def test_rank_mismatch(self):
        """
        Test that (n, 1) input fails even if dimension matches n.
        """
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0] - 1.0
            return func

        n = 2
        pkg_factory = certificate_package(cbf_factory, n=n)
        dummy_conditions = lambda val: val
        cert_collection = pkg_factory(certificate_conditions=dummy_conditions)

        j_func = cert_collection.jacobians[0]

        t = 0.0
        x = jnp.array([[2.0], [3.0]]) # Shape (2, 1)

        with self.assertRaises(ValueError) as cm:
            j_func(t, x)

        self.assertIn("State dimension mismatch", str(cm.exception))
        self.assertIn("expected input shape (2,)", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
