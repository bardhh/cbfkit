"""Zeroing CBF constraints: Lfh + Lgh * u + alpha(h) >= 0."""

from typing import Any, Callable, Tuple

from jax import Array

from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CbfClfQpData,
    CertificateCollection,
    DynamicsCallable,
    State,
    Time,
)

from ._constraint_core import build_cbf_constraint_generator


def generate_compute_zeroing_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    return build_cbf_constraint_generator(control_limits, dyn_func, barriers, lyapunovs, **kwargs)
