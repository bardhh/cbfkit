"""Core builders for CBF and CLF constraint generators.

Extracts the common boilerplate shared across zeroing/robust/stochastic
variants, reducing near-identical code to thin wrappers that specify
only the variant-specific math (extra b-vector terms).
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, lax

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
from .unpack import unpack_for_cbf, unpack_for_clf

# Callable computing variant-specific addition to b_vec.
# Signature: (jacobians, hessians_or_none, state) -> scalar_or_array
ExtraBTermFn = Callable[[Array, Optional[Array], Array], Any]


def batched_hessian_trace(s: Array, hessians: Array) -> Array:
    """Compute 0.5 * Tr[s^T H_i s] for each Hessian H_i using vmap.

    Replaces the Python list comprehension pattern that forces XLA to
    unroll and trace each iteration separately.

    Args:
        s: noise covariance matrix from sigma(x).
        hessians: stacked Hessian matrices (n_certs, n, n).

    Returns:
        Array of trace values, shape (n_certs,).
    """

    def _single_trace(h_i):
        return 0.5 * jnp.trace(jnp.matmul(jnp.matmul(s.T, h_i), s))

    return jax.vmap(_single_trace)(hessians)


def build_cbf_constraint_generator(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    compute_hessians: bool = False,
    extra_b_term_fn: Optional[ExtraBTermFn] = None,
    certificate_package: Optional[CertificateCollection] = None,
    **kwargs: Any,
) -> Callable:
    """Build a standard CBF constraint generator.

    Covers zeroing, robust, stochastic, activated, and consolidated variants.
    The variant-specific math is injected via ``extra_b_term_fn``.

    Args:
        control_limits: Symmetric actuation limits.
        dyn_func: Dynamics callable returning ``(f(x), g(x))``.
        barriers: Barrier certificate collection.
        lyapunovs: Lyapunov certificate collection.
        compute_hessians: Whether certificate Hessians are needed.
        extra_b_term_fn: Variant-specific b_vec term.
            Called as ``extra_b_term_fn(jacobians, hessians, state)``.
            Returns a scalar or array added to the base b_vec.
        certificate_package: Override certificate collection (for consolidated CBF).
        **kwargs: Forwarded to ``unpack_for_cbf``
            (``tunable_class_k``, ``relaxable_cbf``, ``scale_cbf``, etc.).
    """
    certs = certificate_package if certificate_package is not None else barriers
    compute_barrier_values = generate_compute_certificate_values(certs, compute_hessians)
    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable, relaxable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )
    scale_cbf = kwargs.get("scale_cbf", 1.0)

    @jit
    def compute_cbf_constraints(
        t: Time,
        x: State,
        f: Optional[Array] = None,
        g: Optional[Array] = None,
    ) -> Tuple[Array, Array, CbfClfQpData]:
        nonlocal a_cbf, b_cbf
        data: CbfClfQpData = {}

        dyn_f = f
        dyn_g = g
        if dyn_f is None or dyn_g is None:
            dyn_f, dyn_g = dyn_func(x)

        if n_bfs > 0:
            bf_x, bj_x, bh_x, dbf_t, bc_x = compute_barrier_values(t, x)
            extra = extra_b_term_fn(bj_x, bh_x, x) if extra_b_term_fn is not None else 0.0

            a_cbf = a_cbf.at[:, :n_con].set(-jnp.matmul(bj_x, dyn_g))
            b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f) + extra + bc_x)
            if tunable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-scale_cbf * jnp.diag(bc_x))
                b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f) + extra)
            elif relaxable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-scale_cbf * jnp.eye(n_bfs))

            violated = lax.cond(jnp.any(bf_x < 0), lambda _fake: True, lambda _fake: False, 0)
            data["bfs"] = bf_x
            data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints


def build_clf_constraint_generator(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    compute_hessians: bool = False,
    extra_b_term_fn: Optional[ExtraBTermFn] = None,
    additive_relaxation: bool = True,
    strict_completion: bool = False,
    **kwargs: Any,
) -> Callable:
    """Build a standard CLF constraint generator.

    Covers vanilla, robust, and stochastic variants.

    Args:
        control_limits: Symmetric actuation limits.
        dyn_func: Dynamics callable returning ``(f(x), g(x))``.
        barriers: Barrier certificate collection.
        lyapunovs: Lyapunov certificate collection.
        compute_hessians: Whether certificate Hessians are needed.
        extra_b_term_fn: Variant-specific b_vec term (sign-adjusted by caller).
        additive_relaxation: If True, use ``-scale_clf * I`` for relaxation column
            and keep b_vec unchanged. If False, use ``-lc_x`` for relaxation column
            and recompute b_vec without ``lc_x`` or extra term (multiplicative).
        strict_completion: If True, use ``lf_x < 0`` for completion check.
            If False, use ``lf_x <= clf_complete_tol`` from kwargs.
        **kwargs: Forwarded to ``unpack_for_clf``.
    """
    compute_lyapunov_values = generate_compute_certificate_values(lyapunovs, compute_hessians)
    n_con, _n_bfs, n_lfs, a_clf, b_clf, relaxable = unpack_for_clf(
        control_limits, lyapunovs, barriers, **kwargs
    )
    scale_clf = kwargs.get("scale_clf", 1.0)
    clf_complete_tol = kwargs.get("clf_complete_tol", 1e-3)

    @jit
    def compute_clf_constraints(
        t: Time,
        x: State,
        f: Optional[Array] = None,
        g: Optional[Array] = None,
    ) -> Tuple[Array, Array, CbfClfQpData]:
        nonlocal a_clf, b_clf
        data: CbfClfQpData = {}

        dyn_f = f
        dyn_g = g
        if dyn_f is None or dyn_g is None:
            dyn_f, dyn_g = dyn_func(x)

        if n_lfs > 0:
            lf_x, lj_x, lh_x, dlf_t, lc_x = compute_lyapunov_values(t, x)
            extra = extra_b_term_fn(lj_x, lh_x, x) if extra_b_term_fn is not None else 0.0

            a_clf = a_clf.at[:, :n_con].set(jnp.matmul(lj_x, dyn_g))
            b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f) + extra + lc_x)
            if relaxable:
                if additive_relaxation:
                    a_clf = a_clf.at[:, -n_lfs:].set(-scale_clf * jnp.eye(n_lfs))
                else:
                    a_clf = a_clf.at[:, -n_lfs:].set(-lc_x)
                    b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f))

            if strict_completion:
                complete = lax.cond(
                    jnp.all(lf_x < 0),
                    lambda _fake: True,
                    lambda _fake: False,
                    0,
                )
            else:
                complete = lax.cond(
                    jnp.all(lf_x <= clf_complete_tol),
                    lambda _fake: True,
                    lambda _fake: False,
                    0,
                )

            data["lfs"] = lf_x
            data["complete"] = complete

        return a_clf, b_clf, data

    return compute_clf_constraints
