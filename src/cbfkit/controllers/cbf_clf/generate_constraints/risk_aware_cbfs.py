"""Risk-aware CBF constraints.

generate_compute_ra_cbf_constraints: placeholder (theory in development).
generate_compute_estimate_feedback_ra_cbf_constraints: estimate-feedback variant.
"""

import warnings
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax, scipy

from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams

from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CbfClfQpData,
    CertificateCollection,
    DynamicsCallable,
    State,
    Time,
)

from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)
from .unpack import unpack_for_cbf


def generate_compute_ra_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    """Risk-aware CBF constraint generator.

    .. warning::
        Not yet implemented — theory still in development.
        Returns zero constraints (no safety filtering).
        Use ``generate_compute_estimate_feedback_ra_cbf_constraints`` instead.
    """
    warnings.warn(
        "generate_compute_ra_cbf_constraints is not yet implemented "
        "(theory in development). The returned constraints are zero "
        "and provide NO safety filtering. Use "
        "generate_compute_estimate_feedback_ra_cbf_constraints instead.",
        stacklevel=2,
    )

    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable, relaxable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    @jit
    def compute_cbf_constraints(t: Time, x: State) -> Tuple[Array, Array, CbfClfQpData]:
        data: CbfClfQpData = {}
        return a_cbf, b_cbf, data

    return compute_cbf_constraints


####################################################################################################
### RISK-AWARE ESTIMATE-FEEDBACK CBF ################################################################
def generate_compute_estimate_feedback_ra_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    compute_lyapunov_values = generate_compute_certificate_values(lyapunovs)
    n_con, n_bfs, _n_lfs, a_clf, b_clf, tunable, relaxable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    # Check for Risk-Aware Params object
    if "ra_cbf_params" in kwargs:
        ra_params: RiskAwareParams = kwargs["ra_cbf_params"]  # type: ignore[assignment]
        assert ra_params.t_max is not None
        assert ra_params.eta is not None
        assert ra_params.p_bound is not None
        assert ra_params.epsilon is not None
        assert ra_params.lambda_generator is not None
        r_buffer = float(
            jnp.sqrt(2 * ra_params.t_max)
            * ra_params.eta
            * scipy.special.erfinv(2 * ra_params.p_bound - 1)
            + ra_params.epsilon * ra_params.lambda_generator
        )
    else:
        ra_params = RiskAwareParams(sigma=lambda x: jnp.zeros((x.shape[0], 1)))
        r_buffer = 0.0

    @jit
    def compute_clf_constraints(t: Time, x: State) -> Tuple[Array, Array, CbfClfQpData]:
        """Computes CBF and CLF constraints."""
        nonlocal a_clf, b_clf
        data: CbfClfQpData = {}
        dyn_f, dyn_g = dyn_func(x)
        # Get K matrix from kwargs (passed from estimator state)
        k_mat = kwargs.get("kalman_gain", jnp.zeros((x.shape[0], x.shape[0])))

        if n_bfs > 0:
            lf_x, lj_x, lh_x, dlf_t, lc_x = compute_lyapunov_values(t, x)
            assert ra_params.varsigma is not None
            product_varsigma_and_k = jnp.matmul(ra_params.varsigma(x), k_mat)
            traces = jnp.array(
                [
                    0.5
                    * jnp.trace(
                        jnp.matmul(
                            jnp.matmul(product_varsigma_and_k.T, lh_ii), product_varsigma_and_k
                        )
                    )
                    for lh_ii in lh_x
                ]
            )
            assert ra_params.lambda_h is not None
            assert ra_params.epsilon is not None
            estimate_feedback_term = (
                ra_params.lambda_h * ra_params.epsilon * jnp.linalg.norm(jnp.matmul(lj_x, k_mat))
            )

            a_clf = a_clf.at[:, :n_con].set(jnp.matmul(lj_x, dyn_g))
            b_clf = b_clf.at[:].set(
                -dlf_t - jnp.matmul(lj_x, dyn_f) - traces - estimate_feedback_term + lc_x
            )
            if tunable:
                a_clf = a_clf.at[:, -n_bfs:].set(-lc_x)
                b_clf = b_clf.at[:].set(
                    -dlf_t - jnp.matmul(lj_x, dyn_f) - traces - estimate_feedback_term
                )
            elif relaxable:
                a_clf = a_clf.at[:, -n_bfs:].set(-1.0)

            complete = lax.cond(jnp.all(lf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["lfs"] = lf_x
            data["lfs_nom"] = lf_x - r_buffer
            data["complete"] = complete

        return a_clf, b_clf, data

    return compute_clf_constraints
