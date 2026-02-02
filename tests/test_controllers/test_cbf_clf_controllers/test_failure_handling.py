import jax.numpy as jnp
import pytest
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)
from cbfkit.utils.user_types import CertificateCollection, ControllerData


def test_qp_solver_failure_handling():
    """Test that the controller handles QP failures correctly (UNSOLVED/Infeasible)."""

    # 1. Trivial Dynamics (1D)
    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    # 2. Infeasible CBF: h(x) = -1 (always unsafe)
    def cbf(t, x):
        return -1.0  # scalar

    def cbf_grad(t, x):
        return jnp.array([0.0])

    def cbf_hess(t, x):
        return jnp.array([[0.0]])

    def cbf_partial(t, x):
        return 0.0

    def cbf_cond(val):
        return val  # alpha(h) = h. So -1.

    barriers = CertificateCollection([cbf], [cbf_grad], [cbf_hess], [cbf_partial], [cbf_cond])

    # 3. Controller
    setup_controller = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints,
    )

    controller = setup_controller(
        control_limits=jnp.array([1.0]),
        dynamics_func=dynamics,
        barriers=barriers,
        slack_bound_cbf=1e-3,
        relaxable_cbf=False,
    )

    # 4. Execute one step
    t = 0.0
    x = jnp.array([0.0])
    u_nom = jnp.array([0.0])
    key = jnp.array([0, 0], dtype=jnp.uint32)
    data = ControllerData()

    u, new_data = controller(t, x, u_nom, key, data)

    # Check results
    assert new_data.error == True
    assert jnp.all(jnp.isnan(u))
    # Status 0 is UNSOLVED (which is what we got in repro)
    # The actual status returned by solver might be 0 or 3 depending on solver config/luck
    # But we expect failure.
    assert new_data.error_data == 0 or new_data.error_data == 3

    # Check that error propagates
    # Call again with prev error
    u2, new_data2 = controller(t, x, u_nom, key, new_data)
    assert new_data2.error == True
    assert jnp.all(jnp.isnan(u2))
