
import pytest
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints.zeroing_cbfs import generate_compute_zeroing_cbf_constraints
from cbfkit.controllers.cbf_clf.generate_constraints.vanilla_clfs import generate_compute_vanilla_clf_constraints
from cbfkit.utils.user_types import CertificateCollection

def test_certificate_validation_invalid_type():
    """Test that passing a list instead of CertificateCollection raises ValueError."""
    generator = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints
    )

    def dynamics(x):
        return jnp.zeros((2,)), jnp.zeros((2, 1))

    # Should raise ValueError/TypeError because it's a list, not length 5 tuple-like
    # Note: If it's a list of length 1, it hits the length check.
    with pytest.raises(ValueError, match="Expected a CertificateCollection with 5 elements"):
        generator(
            control_limits=jnp.array([1.0]),
            dynamics_func=dynamics,
            barriers=[lambda t, x: 0.0]
        )

def test_certificate_validation_invalid_length():
    """Test that passing a tuple of wrong length raises ValueError."""
    generator = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints
    )

    def dynamics(x):
        return jnp.zeros((2,)), jnp.zeros((2, 1))

    with pytest.raises(ValueError, match="but got a collection of length 1"):
        generator(
            control_limits=jnp.array([1.0]),
            dynamics_func=dynamics,
            barriers=([lambda t, x: 0.0],)
        )

def test_certificate_validation_inconsistent_components():
    """Test that passing a CertificateCollection with inconsistent list lengths raises ValueError."""
    generator = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints
    )

    def dynamics(x):
        return jnp.zeros((2,)), jnp.zeros((2, 1))

    # functions has 1 element, others have 0
    inconsistent_barriers = CertificateCollection(
        [lambda t, x: 0.0], [], [], [], []
    )

    with pytest.raises(ValueError, match="Inconsistent component lengths"):
        generator(
            control_limits=jnp.array([1.0]),
            dynamics_func=dynamics,
            barriers=inconsistent_barriers
        )
