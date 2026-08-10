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
    """Generates zeroing CBF constraints with activation weight scaling.

    Each barrier row (its control columns, its slack column, and its right-hand side) is
    multiplied by that barrier's activation weight. Scaling an inequality by a positive
    constant leaves its feasible set untouched, so the weight is an on/off switch rather
    than a soft knob: a barrier with any weight above zero is enforced exactly as it would
    be unweighted, and only a weight of exactly zero -- which ``compute_activation_weights``
    produces for every obstacle outside the k-closest set -- removes it from the QP.

    Zeroing the whole row is what deactivation means here, so the resulting all-zero row is
    intentional. ``cbf_clf_qp_generator`` recognizes such rows and substitutes an inactive
    regularization row; it must not treat them as a reason to discard the QP solution.
    """
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
        # Scale the slack column along with the rest of the row: leaving it unscaled would
        # divide through to an effective slack of delta/w, amplifying the relaxation of a
        # barely-activated barrier by 1/w.
        a_cbf = a_cbf * weights[:, None]
        b_cbf = b_cbf * weights
        data["activation_weights"] = weights
        return a_cbf, b_cbf, data

    return compute_cbf_constraints
