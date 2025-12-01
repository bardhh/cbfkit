"""
#! docstring
"""

from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax

from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CertificateCollection,
    DynamicsCallable,
    State,
    Time,
)

from ..utils.robustness_terms import robustness_sup_norm, robustness_two_norm
from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)
from .unpack import unpack_for_cbf


####################################################################################################
### ZEROING CBF: Lfh + Lgh * u + \alpha(h) >= 0 ####################################################
def generate_compute_robust_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Dict[str, Any],
) -> Callable[[Time, State], Tuple[Array, Array, Dict[str, Any]]]:
    """
    #! To Do: docstring
    """
    compute_barrier_values = generate_compute_certificate_values(barriers)
    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable, relaxable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    # Check for robustness term
    if "disturbance_norm_bound" not in kwargs:
        raise ValueError("kwargs missing disturbance_norm_bound (float)!")

    if "disturbance_norm" not in kwargs:
        raise ValueError("kwargs missing disturbance_norm (int) (e.g., 2-norm, sup-norm)!")

    disturbance_norm = kwargs["disturbance_norm"]
    disturbance_norm_bound = jnp.array(kwargs["disturbance_norm_bound"])

    if disturbance_norm == 2:
        robustness_term = robustness_two_norm(disturbance_norm_bound)

    elif disturbance_norm == jnp.inf:
        robustness_term = robustness_sup_norm(disturbance_norm_bound)

    @jit
    def compute_cbf_constraints(t: Time, x: State) -> Tuple[Array, Array, Dict[str, Any]]:
        """Computes CBF and CLF constraints."""
        nonlocal a_cbf, b_cbf
        data = {}
        dyn_f, dyn_g = dyn_func(x)

        if n_bfs > 0:
            bf_x, bj_x, _, dbf_t, bc_x = compute_barrier_values(t, x)

            a_cbf = a_cbf.at[:, :n_con].set(-jnp.matmul(bj_x, dyn_g))
            b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f) - robustness_term(bj_x) + bc_x)
            if tunable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-bc_x)
                b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f) - robustness_term(bj_x))
            elif relaxable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-1.0)

            violated = lax.cond(jnp.any(bf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["bfs"] = bf_x
            data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints
