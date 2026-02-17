
import unittest
import jax.numpy as jnp
from cbfkit.certificates import certificate_package

class TestCertificatePackager(unittest.TestCase):
    def test_default_concatenated(self):
        # Default behavior: h(xt)
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0] - 1.0 # x[0] >= 1.0
            return func

        # Use simple lambda for conditions
        dummy_conditions = lambda x: x

        pkg = certificate_package(cbf_factory, n=1)
        collection = pkg(certificate_conditions=dummy_conditions)

        # Check barrier value
        # h(x) = x[0] - 1.
        # x = [2.], t = 0. -> h = 1.
        # Collection functions take (t, x)
        h_val = collection.functions[0](0.0, jnp.array([2.0]))
        self.assertAlmostEqual(float(h_val), 1.0)

    def test_separated_signature(self):
        # New behavior: h(t, x)
        def cbf_factory(**kwargs):
            def func(t, x):
                return x[0] - t # x[0] >= t
            return func

        dummy_conditions = lambda x: x

        # Pass input_style="separated"
        pkg = certificate_package(cbf_factory, n=1, input_style="separated")
        collection = pkg(certificate_conditions=dummy_conditions)

        # x = [2.], t = 1. -> h = 2 - 1 = 1.
        h_val = collection.functions[0](1.0, jnp.array([2.0]))
        self.assertAlmostEqual(float(h_val), 1.0)

        # Check gradient
        # h = x - t
        # grad_x = [1.]
        # partial_t = -1.
        grad_val = collection.jacobians[0](1.0, jnp.array([2.0]))
        self.assertTrue(jnp.allclose(grad_val, jnp.array([1.0])))

        partial_t_val = collection.partials[0](1.0, jnp.array([2.0]))
        self.assertAlmostEqual(float(partial_t_val), -1.0)

    def test_state_signature(self):
        # New behavior: h(x)
        def cbf_factory(**kwargs):
            def func(x):
                return x[0] * 2.0
            return func

        dummy_conditions = lambda x: x

        # Pass input_style="state"
        pkg = certificate_package(cbf_factory, n=1, input_style="state")
        collection = pkg(certificate_conditions=dummy_conditions)

        # x = [2.] -> h = 4.
        h_val = collection.functions[0](0.0, jnp.array([2.0]))
        self.assertAlmostEqual(float(h_val), 4.0)

        # Check gradient
        # grad_x = [2.]
        grad_val = collection.jacobians[0](0.0, jnp.array([2.0]))
        self.assertTrue(jnp.allclose(grad_val, jnp.array([2.0])))

        # partial_t should be 0
        partial_t_val = collection.partials[0](0.0, jnp.array([2.0]))
        self.assertAlmostEqual(float(partial_t_val), 0.0)

    def test_direct_function(self):
        # Test use_factory=False with input_style="state"
        def h(x):
            return x[0] - 1.0

        dummy_conditions = lambda x: x

        # No factory needed!
        pkg = certificate_package(h, n=1, input_style="state", use_factory=False)
        collection = pkg(certificate_conditions=dummy_conditions)

        # h(x) = x[0] - 1.
        # x = [2.] -> h = 1.
        h_val = collection.functions[0](0.0, jnp.array([2.0]))
        self.assertAlmostEqual(float(h_val), 1.0)

        # Gradient should be [1.]
        grad_val = collection.jacobians[0](0.0, jnp.array([2.0]))
        self.assertTrue(jnp.allclose(grad_val, jnp.array([1.0])))

if __name__ == '__main__':
    unittest.main()
