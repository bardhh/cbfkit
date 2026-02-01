"""
#! docstring
"""

from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax, vmap

from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CertificateCollection,
    ComputeCertificateConstraintFunctionGenerator,
    DynamicsCallable,
    State,
)

from ..utils.utils import block_diag_matrix_from_vec, interleave_arrays


####################################################################################################
### Generate Input Constraints #####################################################################
def generate_compute_input_constraints(
    control_limits: Array,
) -> Callable[[float, State], Tuple[Array, Array]]:
    """Generator function for the callable that will compute the input constraints.

    Args:
        control_limits (Array): 1D array of actuation limits (symmetric about origin)

    Returns
    -------
        compute_input_constraints (Callable[[float, State], Tuple[Array, Array]]):
            function to compute input constraints
    """
    a_mat = block_diag_matrix_from_vec(len(control_limits))
    b_vec = interleave_arrays(control_limits, control_limits)

    @jit
    def compute_input_constraints(_t: float, _x: Array) -> Tuple[Array, Array]:
        """Computes input constraints."""
        return a_mat, b_vec

    return compute_input_constraints


####################################################################################################
### Generate CBF and CLF Constraints ###############################################################
def generate_compute_cbf_clf_constraints(
    generate_compute_cbf_constraints: ComputeCertificateConstraintFunctionGenerator,
    generate_compute_clf_constraints: ComputeCertificateConstraintFunctionGenerator,
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs,
) -> Callable[[float, Array], Tuple[Array, Array, Dict[str, Any]]]:
    """_summary_

    Args:
        generate_compute_cbf_constraints (ComputeCertificateConstraintFunctionGenerator): _description_
        generate_compute_clf_constraints (ComputeCertificateConstraintFunctionGenerator): _description_
        control_limits (Array): _description_
        dyn_func (DynamicsCallable): _description_
        barriers (CertificateCollection, optional): _description_. Defaults to ([], [], [], [], []).
        lyapunovs (CertificateCollection, optional): _description_. Defaults to ([], [], [], [], []).

    Returns
    -------
        Callable[[float, Array], Tuple[Array, Array, Dict[str, Any]]]: _description_
    """
    compute_cbf_constraints = generate_compute_cbf_constraints(
        control_limits, dyn_func, barriers, lyapunovs, **kwargs
    )
    compute_clf_constraints = generate_compute_clf_constraints(
        control_limits, dyn_func, barriers, lyapunovs, **kwargs
    )

    @jit
    def compute_cbf_clf_constraints(t: float, x: Array) -> Tuple[Array, Array, Dict[str, Any]]:
        """_summary_

        Returns
        -------
            _type_: _description_
        """
        amat_cbf, bvec_cbf, cbf_data = compute_cbf_constraints(t, x)
        amat_clf, bvec_clf, clf_data = compute_clf_constraints(t, x)

        return (
            jnp.vstack([amat_cbf, amat_clf]),
            jnp.hstack([bvec_cbf, bvec_clf]),
            {**cbf_data, **clf_data},
        )

    return compute_cbf_clf_constraints


def _stack_and_validate(values: list, name: str):
    """Stacks a list of arrays and validates that they all have the same shape."""
    if not values:
        return jnp.stack(values)

    first_shape = jnp.shape(values[0])
    for i, val in enumerate(values):
        val_shape = jnp.shape(val)
        if val_shape != first_shape:
            raise ValueError(
                f"Shape mismatch in {name} at index {i}: "
                f"Expected {first_shape} (consistent with index 0), but got {val_shape}. "
                f"Ensure all {name} return arrays of the same shape."
            )

    return jnp.stack(values)


def generate_compute_certificate_values_list_comprehension(
    certificate_package, compute_hessians: bool = True
):
    functions, jacobians, hessians, partials, conditions = certificate_package

    @jit
    def compute_certificate_values_list_comprehension(t, x):
        bf_values = [lf(t, x) for lf in functions]
        bf_x = _stack_and_validate(bf_values, "certificate functions")

        bj_values = [lj(t, x) for lj in jacobians]
        bj_x = _stack_and_validate(bj_values, "certificate jacobians")

        if compute_hessians:
            bh_values = [lh(t, x) for lh in hessians]
            bh_x = _stack_and_validate(bh_values, "certificate hessians")
        else:
            bh_x = None

        dbf_t_values = [lt(t, x) for lt in partials]
        dbf_t = _stack_and_validate(dbf_t_values, "certificate partials")

        bc_values = [lc(lf) for lc, lf in zip(conditions, bf_x)]
        bc_x = _stack_and_validate(bc_values, "certificate conditions")

        return bf_x, bj_x, bh_x, dbf_t, bc_x

    return compute_certificate_values_list_comprehension


def generate_compute_certificate_values_vmap(
    certificate_package, compute_hessians: bool = True
):
    """
    Computes certificate values using list comprehension (unrolling).

    Note (Bolt): This function was optimized to use list comprehension instead of
    lax.switch inside vmap. For lists of distinct closures (standard in cbfkit),
    vmap+switch introduces overhead without reducing graph size. Unrolling avoids
    dispatch overhead and allows XLA to optimize the concatenated graph effectively.
    """
    return generate_compute_certificate_values_list_comprehension(
        certificate_package, compute_hessians
    )
