"""
robustness_terms.py
================

Contains functions to compute robustness terms for CBF conditions to reject additive, bounded
disturbances to the system dynamics.

Functions
---------
-robustness_two_norm(bound): computes the robustness term for a perturbation with a known bound on the 2-norm
-robustness_sup_norm(bound): computes the robustness term for a perturbation with a known bound on the supremum (inf)-norm

Notes
-----
Used in the creation of Robust CBF/CLF control laws.

Examples
--------
>>> from cbfkit.controllers.cbf_clf.utils.robustness_terms import robustness_two_norm
>>> disturbance_two_norm_bound = 10.0
>>> cbf_protection_function = robustness_two_norm(disturbance_two_norm_bound)

"""

from typing import Callable

import jax.numpy as jnp
from jax import Array


def robustness_two_norm(bound: Array) -> Callable[[Array], Array]:
    """Compute robustness term (2-norm) in CBF condition.

    Args:
        bound (Array): upper bound on 2-norm of disturbance within operating domain

    Returns
    -------
        Array: value of robustness term
    """

    def compute(jacobian: Array) -> Array:
        """Compute robustness term.

        Args:
            jacobian (Array): dhdx

        Returns
        -------
            Array: value
        """
        # Row-wise: ``jacobian`` is the (n_certificates x n) stack of dh/dx; each
        # constraint gets its own margin ||dh_i/dx|| * bound. (A 1-D input -- a
        # single certificate -- gives the same scalar as before.)
        return jnp.linalg.norm(jacobian, axis=-1) * bound

    return compute


def robustness_sup_norm(bound: Array) -> Callable[[Array], Array]:
    """Compute robustness term (sup-norm) in CBF condition.

    Args:
        bound (Array): upper bound on sup-norm of disturbance within operating domain

    Returns
    -------
        Array: value of robustness term
    """

    def compute(jacobian: Array) -> Array:
        """Compute robustness term.

        Args:
            jacobian (Array): dhdx

        Returns
        -------
            Array: value
        """
        return jnp.sum(jnp.abs(jacobian * bound), axis=-1)  # row-wise, see robustness_two_norm

    return compute
