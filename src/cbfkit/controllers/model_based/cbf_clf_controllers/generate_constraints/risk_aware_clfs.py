"""
#! docstring
"""

from typing import Callable, Tuple, Dict, Any
import jax.numpy as jnp
from jax import Array, jit, lax, scipy
from cbfkit.utils.user_types import (
    DynamicsCallable,
    CertificateCollection,
    State,
)
from .unpack import unpack_for_clf
from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)
from cbfkit.estimators.kalman_filters.ekf import get_global_k_ekf
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.risk_aware_params import (
    RiskAwareParams,
)


####################################################################################################
### RISK-AWARE CLF: LfV + LgV*u + 0.5*Tr[sigma.T * d2V/dx2 * sigma] <= c(Vr) #######################
def generate_compute_ra_clf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = ([], [], [], [], []),
    lyapunovs: CertificateCollection = ([], [], [], [], []),
    **kwargs: Dict[str, Any],
) -> Callable[[float, State], Tuple[Array, Array, Dict[str, Any]]]:
    """
    #! To Do: docstring
    """
    compute_lyapunov_values = generate_compute_certificate_values(lyapunovs)
    n_con, _n_bfs, n_lfs, a_clf, b_clf, relaxable = unpack_for_clf(
        control_limits, lyapunovs, barriers, **kwargs
    )

    # Check for Risk-Aware Params object
    if "ra_clf_params" in kwargs:
        ra_params = kwargs["ra_clf_params"]
        r_buffer = float(
            jnp.sqrt(2 * ra_params.t_max)
            * ra_params.eta
            * scipy.special.erfinv(2 * ra_params.p_bound - 1)
        )
    else:
        ra_params = RiskAwareParams(sigma=lambda _: None)
        r_buffer = 0.0

    @jit
    def compute_clf_constraints(t: float, x: State) -> Tuple[Array, Array]:
        """Computes CBF and CLF constraints."""
        nonlocal a_clf, b_clf
        data = {}
        dyn_f, dyn_g = dyn_func(x)
        sigma = ra_params.sigma(x)

        if n_lfs > 0:
            lf_x, lj_x, lh_x, dlf_t, lc_x = compute_lyapunov_values(t, x)
            traces = jnp.array(
                [0.5 * jnp.trace(jnp.matmul(jnp.matmul(sigma.T, lh_ii), sigma)) for lh_ii in lh_x]
            )

            a_clf = a_clf.at[:, :n_con].set(jnp.matmul(lj_x, dyn_g))
            b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f) - traces + lc_x)
            if relaxable:
                a_clf = a_clf.at[:, -n_lfs:].set(-lc_x)
                b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f) - traces)

            complete = lax.cond(jnp.all(lf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["lfs"] = lf_x
            data["lfs_nom"] = lf_x - r_buffer
            data["complete"] = complete

        return a_clf, b_clf, data

    return compute_clf_constraints


####################################################################################################
### RISK-AWARE CLF: LfV + LgV*u + 0.5*Tr[sigma.T * d2V/dx2 * sigma] <= c(Vr) #######################
def generate_compute_estimate_feedback_ra_clf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = ([], [], [], [], []),
    lyapunovs: CertificateCollection = ([], [], [], [], []),
    **kwargs: Dict[str, Any],
) -> Callable[[float, State], Tuple[Array, Array, Dict[str, Any]]]:
    """
    #! To Do: docstring
    """
    compute_lyapunov_values = generate_compute_certificate_values(lyapunovs)
    n_con, _n_bfs, n_lfs, a_clf, b_clf, relaxable = unpack_for_clf(
        control_limits, lyapunovs, barriers, **kwargs
    )

    # Check for Risk-Aware Params object
    if "ra_clf_params" in kwargs:
        ra_params = kwargs["ra_clf_params"]
        r_buffer = float(
            jnp.sqrt(2 * ra_params.t_max)
            * ra_params.eta
            * scipy.special.erfinv(2 * ra_params.p_bound - 1)
            + ra_params.epsilon * ra_params.lambda_generator
        )
    else:
        ra_params = RiskAwareParams(sigma=lambda _: None)
        r_buffer = 0.0

    @jit
    def compute_clf_constraints(t: float, x: State) -> Tuple[Array, Array]:
        """Computes CBF and CLF constraints."""
        nonlocal a_clf, b_clf
        data = {}
        dyn_f, dyn_g = dyn_func(x)
        k_mat = get_global_k_ekf()

        if n_lfs > 0:
            lf_x, lj_x, lh_x, dlf_t, lc_x = compute_lyapunov_values(t, x)
            product_varsigma_and_k = jnp.matmul(ra_params.varsigma, k_mat)
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
            estimate_feedback_term = (
                ra_params.lambda_h * ra_params.epsilon * jnp.linalg.norm(jnp.matmul(lj_x, k_mat))
            )

            a_clf = a_clf.at[:, :n_con].set(jnp.matmul(lj_x, dyn_g))
            b_clf = b_clf.at[:].set(
                -dlf_t - jnp.matmul(lj_x, dyn_f) - traces - estimate_feedback_term + lc_x
            )
            if relaxable:
                a_clf = a_clf.at[:, -n_lfs:].set(-lc_x)
                b_clf = b_clf.at[:].set(
                    -dlf_t - jnp.matmul(lj_x, dyn_f) - traces - estimate_feedback_term
                )

            complete = lax.cond(jnp.all(lf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["lfs"] = lf_x
            data["lfs_nom"] = lf_x - r_buffer
            data["complete"] = complete

        return a_clf, b_clf, data

    return compute_clf_constraints
