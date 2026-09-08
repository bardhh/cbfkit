"""The robust CBF/CLF margin must be per certificate, not the norm of the stacked Jacobians.

Regression: with N >= 2 barriers ``robustness_two_norm`` took ``jnp.linalg.norm`` of the
(N x n) stack -- a Frobenius norm over *all* barriers -- and subtracted that one scalar
from every constraint. One barrier was unaffected; the G1 plaza example (5 barriers)
became infeasible at a bound the single-barrier navigate example handled easily.
"""

import jax
import jax.numpy as jnp
import pytest

from cbfkit.controllers.cbf_clf import robust_cbf_clf_qp_controller
from cbfkit.controllers.cbf_clf.utils.robustness_terms import (
    robustness_sup_norm,
    robustness_two_norm,
)
from cbfkit.utils.user_types import CertificateCollection, ControllerData

KEY = jax.random.PRNGKey(0)


def _single_integrator(x):
    return jnp.zeros(2), jnp.eye(2)


def _disc_barrier(center, radius=1.0):
    c = jnp.asarray(center, dtype=float)
    return CertificateCollection(
        [lambda t, x: jnp.sum((x - c) ** 2) - radius**2],
        [lambda t, x: 2.0 * (x - c)],
        [lambda t, x: 2.0 * jnp.eye(2)],
        [lambda t, x: 0.0],
        [lambda val: val],
    )


def _concat(*cs):
    return CertificateCollection(
        *[sum((list(getattr(c, f)) for c in cs), []) for f in cs[0]._fields]
    )


def test_robustness_terms_are_row_wise_for_stacked_jacobians():
    J = jnp.array([[3.0, 4.0], [0.0, 1.0]])
    assert jnp.allclose(robustness_two_norm(jnp.array(0.5))(J), jnp.array([2.5, 0.5]))
    assert jnp.allclose(robustness_sup_norm(jnp.array(0.5))(J), jnp.array([3.5, 0.5]))
    # 1-D input (a single certificate) still gives the scalar it always did
    assert float(robustness_two_norm(jnp.array(0.5))(J[0])) == pytest.approx(2.5)


def test_far_barriers_do_not_tighten_the_active_one():
    """Adding barriers that are far away (huge |dh/dx|) must not change the robust QP's answer."""
    x = jnp.array([1.6, 0.0])  # 0.6 m outside the disc at the origin, moving straight at it
    u_nom = jnp.array([-1.0, 0.0])
    kw = dict(
        control_limits=jnp.array([2.0, 2.0]),
        dynamics_func=_single_integrator,
        disturbance_norm=2,
        disturbance_norm_bound=0.3,
    )
    near = _disc_barrier([0.0, 0.0])
    far = _concat(near, _disc_barrier([40.0, 0.0]), _disc_barrier([0.0, -60.0]))
    u_near, d1 = robust_cbf_clf_qp_controller(barriers=near, **kw)(
        0.0, x, u_nom, KEY, ControllerData()
    )
    u_far, d2 = robust_cbf_clf_qp_controller(barriers=far, **kw)(
        0.0, x, u_nom, KEY, ControllerData()
    )
    assert not bool(d1.error) and not bool(d2.error)
    assert float(u_near[0]) > float(u_nom[0])  # the near barrier does brake
    assert jnp.allclose(u_near, u_far, atol=1e-4), (u_near, u_far)


def test_estimate_feedback_risk_aware_margin_is_per_certificate():
    """Same bug class in the risk-aware estimate-feedback generators: ``||dh/dx K||`` must be
    taken per certificate row, so the b-vector of a two-certificate stack equals the two
    single-certificate b-vectors side by side."""
    from cbfkit.controllers.cbf_clf.generate_constraints.risk_aware_cbfs import (
        generate_compute_estimate_feedback_ra_cbf_constraints,
    )
    from cbfkit.controllers.cbf_clf.generate_constraints.risk_aware_clfs import (
        generate_compute_estimate_feedback_ra_clf_constraints,
    )
    from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams

    def dyn(x):
        return jnp.zeros(2), jnp.eye(2)

    def cert(scale, idx):
        return CertificateCollection(
            [lambda t, x: scale * x[idx] - 0.1],
            [lambda t, x: scale * jnp.eye(2)[idx]],
            [lambda t, x: jnp.zeros((2, 2))],
            [lambda t, x: 0.0],
            [lambda val: val],
        )

    params = RiskAwareParams(
        t_max=10.0,
        p_bound=0.95,
        eta=2.0,
        epsilon=0.1,
        lambda_h=1.0,
        lambda_generator=1.0,
        gamma=jnp.zeros(1),
        sigma=lambda x: 0.25 * jnp.eye(2),
        varsigma=lambda x: 0.1 * jnp.eye(2),
    )
    c1, c2 = cert(1.0, 0), cert(7.0, 1)
    both = _concat(c1, c2)
    x, k = jnp.array([1.0, 2.0]), jnp.eye(2)
    limits = jnp.array([1.0, 1.0])
    for gen, key in (
        (generate_compute_estimate_feedback_ra_cbf_constraints, "ra_cbf_params"),
        (generate_compute_estimate_feedback_ra_clf_constraints, "ra_clf_params"),
    ):
        kw = {key: params, "kalman_gain": k}
        _, b_both, _ = gen(limits, dyn, both, both, **kw)(0.0, x)
        _, b1, _ = gen(limits, dyn, c1, c1, **kw)(0.0, x)
        _, b2, _ = gen(limits, dyn, c2, c2, **kw)(0.0, x)
        assert jnp.allclose(b_both, jnp.concatenate([b1, b2])), (gen.__name__, b_both, b1, b2)
