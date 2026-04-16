"""Activated CBF constraints: zeroing CBF scaled by proximity-based activation weights."""

from typing import Any, Callable, Tuple

from jax import Array, jit

from cbfkit.controllers.cbf_clf.utils.barrier_activation import compute_activation_weights
from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CbfClfQpData,
    CertificateCollection,
    DynamicsCallable,
    State,
    Time,
)

from ._constraint_core import build_cbf_constraint_generator


def generate_compute_activated_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    """Generates zeroing CBF constraints with activation weight scaling."""
    obstacle_positions = kwargs.get("obstacle_positions")
    k_closest = kwargs.get("k_closest", 3)
    activation_radius = kwargs.get("activation_radius", 2.0)
    activation_smoothness = kwargs.get("activation_smoothness", 5.0)

    base_fn = build_cbf_constraint_generator(
        control_limits, dyn_func, barriers, lyapunovs, **kwargs
    )

    if obstacle_positions is None:
        return base_fn

    @jit
    def compute_cbf_constraints(t, x, f=None, g=None):
        a_cbf, b_cbf, data = base_fn(t, x, f=f, g=g)
        weights = compute_activation_weights(
            x,
            obstacle_positions,
            k=k_closest,
            radius=activation_radius,
            smoothness=activation_smoothness,
        )
        a_cbf = a_cbf * weights[:, None]
        b_cbf = b_cbf * weights
        data["activation_weights"] = weights
        return a_cbf, b_cbf, data

    return compute_cbf_constraints
