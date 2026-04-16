"""Stochastic CLF constraints: LfV + LgV*u + 0.5*Tr[sigma.T * d2V/dx2 * sigma] <= -gamma(V)."""

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

from ._constraint_core import batched_hessian_trace, build_clf_constraint_generator


def _make_stochastic_clf_term(sigma):
    """Create extra b-term for stochastic CLF (negative trace of noise-weighted Hessian)."""

    def term(_lj_x, lh_x, x):
        return -batched_hessian_trace(sigma(x), lh_x)

    return term


def generate_compute_stochastic_clf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    if "sigma" not in kwargs:
        raise ValueError("kwargs missing sigma (Callable[[Array], Array])!")
    sigma = kwargs["sigma"]
    if not callable(sigma):
        raise ValueError("sigma must be of type Callable[[Array], Array]!")

    return build_clf_constraint_generator(
        control_limits,
        dyn_func,
        barriers,
        lyapunovs,
        compute_hessians=True,
        extra_b_term_fn=_make_stochastic_clf_term(sigma),
        additive_relaxation=False,
        strict_completion=True,
        **kwargs,
    )
