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


def generate_compute_certificate_values_list_comprehension(
    certificate_package, compute_hessians: bool = True
):
    functions, jacobians, hessians, partials, conditions = certificate_package

    @jit
    def compute_certificate_values_list_comprehension(t, x):
        bf_x = jnp.stack([lf(t, x) for lf in functions])
        bj_x = jnp.stack([lj(t, x) for lj in jacobians])
        if compute_hessians:
            bh_x = jnp.stack([lj(t, x) for lj in hessians])
        else:
            bh_x = None
        dbf_t = jnp.stack([lt(t, x) for lt in partials])
        bc_x = jnp.stack([lc(lf) for lc, lf in zip(conditions, bf_x)])

        return bf_x, bj_x, bh_x, dbf_t, bc_x

    return compute_certificate_values_list_comprehension


def generate_compute_certificate_values_vmap(
    certificate_package, compute_hessians: bool = True
):
    functions, jacobians, hessians, partials, conditions = certificate_package
    # Optimization (Bolt): Use in_axes=(0, None, None) to avoid tiling t and x
    # This avoids O(N) memory allocation and copy for state/time broadcasting
    vmap_bf = vmap(
        lambda i, t, x: lax.switch(i, functions, t, x), in_axes=(0, None, None)
    )
    vmap_bj = vmap(
        lambda i, t, x: lax.switch(i, jacobians, t, x), in_axes=(0, None, None)
    )
    if compute_hessians:
        vmap_bh = vmap(
            lambda i, t, x: lax.switch(i, hessians, t, x), in_axes=(0, None, None)
        )
    vmap_bt = vmap(
        lambda i, t, x: lax.switch(i, partials, t, x), in_axes=(0, None, None)
    )
    vmap_bc = vmap(lambda i, bf: lax.switch(i, conditions, bf))
    index = jnp.arange(len(functions))

    @jit
    def compute_certificate_values_vmap(t, x):
        bf_x = vmap_bf(index, t, x)
        bj_x = vmap_bj(index, t, x)
        if compute_hessians:
            bh_x = vmap_bh(index, t, x)
        else:
            bh_x = None
        dbf_t = vmap_bt(index, t, x)
        bc_x = vmap_bc(index, bf_x)

        return bf_x, bj_x, bh_x, dbf_t, bc_x

    return compute_certificate_values_vmap
