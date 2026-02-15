"""
Test Module for manual gradient handling in cbfkit.certificates.packager

Ensures that manual gradients provided with input_style="state" or "separated"
receive the correct arguments (x or (t, x)) and not the concatenated xt vector.
"""

import unittest
import jax.numpy as jnp
from cbfkit.certificates.packager import certificate_package, CertificateInputStyle

class TestManualGradients(unittest.TestCase):
    def test_state_input_style_manual_grad(self):
        """
        Reproduces a bug where manual gradient function in certificate_package
        receives concatenated [x, t] even if input_style='state', causing
        incorrect gradients if the gradient function relies on input shape/indexing.
        """
        n = 2

        # Define a certificate function h(x) = x[-1] (the last state variable)
        # x = [x0, x1]. h(x) = x1.
        # grad_h(x) = [0, 1].

        def cbf_factory(**kwargs):
            def func(x):
                return x[-1]
            return func

        # Manual gradient factory
        def grad_factory(**kwargs):
            def grad_func(x):
                # Intended behavior: return gradient w.r.t x
                # If x is [x0, x1], we want [0, 1].
                # Using zeros_like(x) makes it depend on input shape.
                # If x is [x0, x1, t], zeros_like(x) is size 3.
                # Then g[-1] sets index 2 (t) to 1.
                # Then slice [:2] returns [0, 0]. Wrong.
                g = jnp.zeros_like(x)
                g = g.at[-1].set(1.0)
                return g
            return grad_func

        # Create package with input_style="state"
        # This implies the user writes func and grad_func in terms of 'x'.
        pkg_factory = certificate_package(
            cbf_factory,
            func_grad=grad_factory,
            n=n,
            input_style="state",  # or CertificateInputStyle.STATE
            use_factory=True
        )

        # Dummy conditions
        dummy_conditions = lambda val: val

        cert = pkg_factory(certificate_conditions=dummy_conditions)
        j_func = cert.jacobians[0]
        t_func = cert.partials[0]

        # Test point
        t = 0.0
        x = jnp.array([10.0, 20.0]) # x1 = 20.

        # Compute gradient
        # Expected: [0.0, 1.0]
        grad_val = j_func(t, x)

        # Check gradient
        self.assertTrue(jnp.allclose(grad_val, jnp.array([0.0, 1.0])),
                        f"Gradient mismatch! Got {grad_val}, expected [0.0, 1.0]")

        # Check partial t (should be 0 for state-only function)
        # If grad_func returns size 2, t_func (which accesses [-1] of result of j_func(xt))
        # Wait, if j_func returns size 2, j_func(xt) returns size 2.
        # Then t_ slices [-1] of size 2 vector? No, j_func returns grad_x.
        # If we fix it, j_func wrapper returns [grad_x, 0].
        # So t_ gets 0.
        pt = t_func(t, x)
        self.assertAlmostEqual(pt, 0.0, places=5)

    def test_separated_input_style_manual_grad(self):
        """
        Tests input_style='separated' with manual gradient.
        Ensures grad function receives (t, x).
        """
        n = 1
        # h(t, x) = t * x[0]
        # grad_x = t
        # grad_t = x[0]

        def cbf_factory(**kwargs):
            def func(t, x):
                return t * x[0]
            return func

        def grad_factory(**kwargs):
            def grad_func(t, x):
                # Returns [grad_x, grad_t]
                # If it receives xt (concatenated), t is xt[-1], x is xt[:-1].
                # If we pass xt to this function expecting (t, x), it fails (arg count mismatch).
                return jnp.array([t, x[0]])
            return grad_func

        pkg_factory = certificate_package(
            cbf_factory,
            func_grad=grad_factory,
            n=n,
            input_style="separated",
            use_factory=True
        )

        cert = pkg_factory(certificate_conditions=lambda v: v)
        j_func = cert.jacobians[0]
        t_func = cert.partials[0]

        t = 2.0
        x = jnp.array([3.0])

        # grad_x = 2.0
        # grad_t = 3.0

        # If not fixed, calling j_func(t, x) calls grad_func(xt) -> TypeError (1 arg instead of 2)
        try:
            gx = j_func(t, x)
            pt = t_func(t, x)

            self.assertAlmostEqual(gx[0], 2.0, places=5)
            self.assertAlmostEqual(pt, 3.0, places=5)

        except TypeError as e:
            self.fail(f"Handling of separated manual gradient failed with TypeError: {e}")
        except Exception as e:
            self.fail(f"Handling of separated manual gradient failed: {e}")

if __name__ == '__main__':
    unittest.main()
