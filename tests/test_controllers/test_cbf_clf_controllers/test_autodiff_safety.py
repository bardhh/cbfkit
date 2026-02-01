
import unittest
import jax
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.certificates import certificate_package, concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.utils.user_types import ControllerData, EMPTY_CERTIFICATE_COLLECTION

class TestAutodiffSafety(unittest.TestCase):
    def test_gradient_at_zero_actuation(self):
        """
        Test that differentiating the controller through a state where
        actuation vanishes (Lg Lf^k h = 0) does not produce NaNs.

        System: x_dot = x * u
        Barrier: h(x) = x  (keep x >= 0)

        Constraint: dot_h + alpha*h >= 0
                   x*u + x >= 0
                   -x*u <= x

        At x=0, the constraint becomes 0*u <= 0.
        The row in A matrix is [0].
        Normalization of this row involves norm(0).
        """

        # 1. Define Dynamics: x_dot = x * u
        def dynamics(x):
            # f(x) = 0
            f = jnp.zeros((1,))
            # g(x) = x (reshaped to 1x1)
            g = x.reshape(1, 1)
            return f, g

        # 2. Define Barrier: h(x) = x
        def cbf_factory(**kwargs):
            def func(xt):
                return xt[0]
            return func

        # Use certificate_package with auto-diff
        package = certificate_package(cbf_factory, n=1)
        # alpha(h) = h
        barriers = concatenate_certificates(
            package(certificate_conditions=linear_class_k(1.0))
        )

        # 3. Setup Controller
        control_limits = jnp.array([10.0]) # u in [-10, 10]

        controller = vanilla_cbf_clf_qp_controller(
            control_limits=control_limits,
            dynamics_func=dynamics,
            barriers=barriers,
            lyapunovs=EMPTY_CERTIFICATE_COLLECTION,
            relaxable_cbf=False
        )

        # 4. Define Loss Function on Controller Output
        def loss_fn(x_state):
            t = 0.0
            u_nom = jnp.array([1.0])
            key = jnp.array([0, 0], dtype=jnp.uint32)
            data = ControllerData()

            # Run controller
            u, _ = controller(t, x_state, u_nom, key, data)

            # Loss is just sum of control input
            return jnp.sum(u)

        # 5. Compute Gradient at x=0
        # At x=0, g(x)=0, so g_mat_c is 0.
        x_target = jnp.array([0.0])

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(x_target)

        print(f"Computed Gradient: {grad}")

        # 6. Assert Validity
        self.assertFalse(jnp.any(jnp.isnan(grad)), "Gradient contains NaNs!")

if __name__ == "__main__":
    unittest.main()
