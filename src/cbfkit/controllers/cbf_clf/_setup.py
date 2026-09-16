"""Static validation for CBF/CLF QP construction (outside traced solves)."""

from typing import Optional

import jax.numpy as jnp
from jax import Array

from cbfkit.certificates import concatenate_certificates
from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CertificateCollection,
    CertificateInput,
)


def _normalize_certificate_collection(
    cert_collection: Optional[CertificateInput], name: str
) -> CertificateCollection:
    """Validates and normalizes certificate collection structure.

    Accepts:
    - None (returns empty collection)
    - CertificateCollection (returns as is)
    - List[CertificateCollection] (concatenates and returns)
    - Tuple/List of length 5 (wraps in CertificateCollection)
    """
    if cert_collection is None:
        return EMPTY_CERTIFICATE_COLLECTION

    collection = None

    # Case 1: Already a CertificateCollection
    if isinstance(cert_collection, CertificateCollection):
        collection = cert_collection

    # Case 2: List/Tuple of CertificateCollections (user passed [c1, c2] or [])
    elif isinstance(cert_collection, (list, tuple)):
        if len(cert_collection) == 0:
            return EMPTY_CERTIFICATE_COLLECTION
        if isinstance(cert_collection[0], CertificateCollection):
            collection = concatenate_certificates(*cert_collection)

    if collection is None:
        # Case 3: Raw tuple of length 5 (legacy structure: (funcs, jacs, hess, parts, conds))
        # Check if it's iterable
        try:
            iter(cert_collection)
        except TypeError:
            raise TypeError(
                f"'{name}' must be a CertificateCollection (tuple/list of length 5) or a list of them, but got {type(cert_collection)}."
            )

        # Check length
        if len(cert_collection) != 5:
            raise ValueError(
                f"Invalid structure for '{name}'. Expected a CertificateCollection with 5 elements "
                "(functions, jacobians, hessians, partials, conditions), "
                f"but got a collection of length {len(cert_collection)}. "
                "Did you pass a list of barrier functions directly? You must provide the derivatives and conditions as well, "
                "or use a helper like 'cbfkit.certificates.CertificateCollection'."
            )

        collection = CertificateCollection(*cert_collection)

    # Validate component consistency
    # Ensure all component lists have the same length to prevent obscure JAX errors downstream.
    lengths = [len(x) for x in collection]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Inconsistent component lengths in '{name}'. "
            f"All components (functions, jacobians, hessians, partials, conditions) must have the same length. "
            f"Got lengths: functions={lengths[0]}, jacobians={lengths[1]}, hessians={lengths[2]}, "
            f"partials={lengths[3]}, conditions={lengths[4]}."
        )

    return collection


def validate_configuration(
    control_limits: Array,
    slack_penalty_cbf: float,
    slack_penalty_clf: float,
    slack_bound_cbf: Optional[float],
    slack_bound_clf: float,
) -> None:
    # Validate configuration to prevent silent failures (e.g., NaNs from negative penalties)
    if slack_penalty_cbf < 0:
        raise ValueError(
            f"Invalid configuration: 'slack_penalty_cbf' must be non-negative, but got {slack_penalty_cbf}."
        )
    if slack_penalty_clf < 0:
        raise ValueError(
            f"Invalid configuration: 'slack_penalty_clf' must be non-negative, but got {slack_penalty_clf}."
        )
    if slack_bound_cbf is not None and slack_bound_cbf <= 0:
        raise ValueError(
            f"Invalid configuration: 'slack_bound_cbf' must be positive, but got {slack_bound_cbf}."
        )
    if slack_bound_clf <= 0:
        raise ValueError(
            f"Invalid configuration: 'slack_bound_clf' must be positive, but got {slack_bound_clf}."
        )
    if jnp.any(jnp.asarray(control_limits) < 0):
        raise ValueError(
            f"Invalid configuration: 'control_limits' elements must be non-negative (defining symmetric bounds |u|<=limit), "
            f"but got {control_limits}."
        )
