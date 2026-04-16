"""Robust CLF constraints: LfV + LgV*u + robustness_term(dV/dx) <= -gamma(V)."""

from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array

from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CbfClfQpData,
    CertificateCollection,
    DynamicsCallable,
    State,
    Time,
)

from ..utils.robustness_terms import robustness_sup_norm, robustness_two_norm
from ._constraint_core import build_clf_constraint_generator


def generate_compute_robust_clf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    if "disturbance_norm_bound" not in kwargs:
        raise ValueError("kwargs missing disturbance_norm_bound (float)!")
    if "disturbance_norm" not in kwargs:
        raise ValueError("kwargs missing disturbance_norm (int) (e.g., 2-norm, sup-norm)!")

    disturbance_norm = kwargs["disturbance_norm"]
    bound = jnp.array(kwargs["disturbance_norm_bound"])

    if disturbance_norm == 2:
        robustness_term = robustness_two_norm(bound)
    elif disturbance_norm == jnp.inf:
        robustness_term = robustness_sup_norm(bound)

    return build_clf_constraint_generator(
        control_limits,
        dyn_func,
        barriers,
        lyapunovs,
        extra_b_term_fn=lambda lj_x, _lh, _x: robustness_term(lj_x),
        additive_relaxation=False,
        strict_completion=True,
        **kwargs,
    )
