
import pytest
import jax.numpy as jnp
from jax import random

from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints.zeroing_cbfs import generate_compute_zeroing_cbf_constraints
from cbfkit.controllers.cbf_clf.generate_constraints.vanilla_clfs import generate_compute_vanilla_clf_constraints
from cbfkit.utils.user_types import CertificateCollection, ControllerData

class TestDynamicsShapeValidation:

    @pytest.fixture
    def controller_gen(self):
        return cbf_clf_qp_generator(
            generate_compute_zeroing_cbf_constraints,
            generate_compute_vanilla_clf_constraints
        )

    @pytest.fixture
    def barriers(self):
        def h(t, x): return x[0]
        def grad_h(t, x): return jnp.array([1.0, 0.0])
        def hess_h(t, x): return jnp.zeros((2, 2))
        def partial_h(t, x): return 0.0
        def class_k(h): return h
        return CertificateCollection(
            [h], [grad_h], [hess_h], [partial_h], [class_k]
        )

    def test_bad_f_shape(self, controller_gen, barriers):
        """Test that f as column vector (n, 1) raises ValueError."""
        def bad_dynamics_f_shape(x):
            f = jnp.array([[x[1]], [0.0]]) # (2, 1) - BAD
            g = jnp.array([[0.0], [1.0]])  # (2, 1) - OK
            return f, g

        control_limits = jnp.array([1.0])
        controller = controller_gen(control_limits, bad_dynamics_f_shape, barriers)

        t, x = 0.0, jnp.array([1.0, 0.0])
        u_nom = jnp.array([0.0])
        key = random.PRNGKey(0)
        data = ControllerData(error=False, error_data=0)

        with pytest.raises(ValueError, match="Dynamics drift term 'f' must be a 1D array"):
            controller(t, x, u_nom, key, data)

    def test_bad_g_shape(self, controller_gen, barriers):
        """Test that g as 1D array (n,) raises ValueError."""
        def bad_dynamics_g_shape(x):
            f = jnp.array([x[1], 0.0])    # (2,) - OK
            g = jnp.array([0.0, 1.0])     # (2,) - BAD (should be 2D)
            return f, g

        control_limits = jnp.array([1.0])
        controller = controller_gen(control_limits, bad_dynamics_g_shape, barriers)

        t, x = 0.0, jnp.array([1.0, 0.0])
        u_nom = jnp.array([0.0])
        key = random.PRNGKey(0)
        data = ControllerData(error=False, error_data=0)

        with pytest.raises(ValueError, match="Dynamics control term 'g' must be a 2D array"):
            controller(t, x, u_nom, key, data)

    def test_mismatched_state_dims(self, controller_gen, barriers):
        """Test that f and g with different state dims raises ValueError."""
        def bad_dynamics_mismatch(x):
            f = jnp.array([x[1], 0.0])          # 2 states
            g = jnp.array([[0.0], [1.0], [0.0]]) # 3 states
            return f, g

        control_limits = jnp.array([1.0])
        controller = controller_gen(control_limits, bad_dynamics_mismatch, barriers)

        t, x = 0.0, jnp.array([1.0, 0.0])
        u_nom = jnp.array([0.0])
        key = random.PRNGKey(0)
        data = ControllerData(error=False, error_data=0)

        with pytest.raises(ValueError, match="State dimension mismatch"):
            controller(t, x, u_nom, key, data)

    def test_mismatched_control_dims(self, controller_gen, barriers):
        """Test that g columns != n_controls raises ValueError."""
        def bad_dynamics_control_mismatch(x):
            f = jnp.array([x[1], 0.0])
            g = jnp.array([[0.0, 0.0], [1.0, 1.0]]) # 2 controls
            return f, g

        control_limits = jnp.array([1.0]) # Expects 1 control
        controller = controller_gen(control_limits, bad_dynamics_control_mismatch, barriers)

        t, x = 0.0, jnp.array([1.0, 0.0])
        u_nom = jnp.array([0.0])
        key = random.PRNGKey(0)
        data = ControllerData(error=False, error_data=0)

        with pytest.raises(ValueError, match="Control dimension mismatch"):
            controller(t, x, u_nom, key, data)
