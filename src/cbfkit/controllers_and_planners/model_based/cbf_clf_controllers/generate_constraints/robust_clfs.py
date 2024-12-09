"""
#! docstring
"""

from typing import Callable, Tuple, Dict, Any
import jax.numpy as jnp
from jax import Array, jit, lax
from cbfkit.utils.user_types import (
    DynamicsCallable,
    CertificateCollection,
    State,
)
from .unpack import unpack_for_clf
from ..utils.robustness_terms import robustness_two_norm, robustness_sup_norm
from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)


####################################################################################################
### VANILLA CLF: LfV + LgV*u <= conditions #########################################################
def generate_compute_robust_clf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = ([], [], [], [], []),
    lyapunovs: CertificateCollection = ([], [], [], [], []),
    **kwargs: Dict[str, Any],
) -> Callable[[float, State], Tuple[Array, Array, Dict[str, Any]]]:
    """
    #! To Do: docstring
    """
    compute_lyapunov_values = generate_compute_certificate_values(lyapunovs)
    n_con, _n_bfs, n_lfs, a_clf, b_clf, relaxable = unpack_for_clf(
        control_limits, lyapunovs, barriers, **kwargs
    )

    # Check for robustness term
    if "disturbance_norm_bound" not in kwargs:
        raise ValueError("kwargs missing disturbance_norm_bound (float)!")

    if "disturbance_norm" not in kwargs:
        raise ValueError("kwargs missing disturbance_norm (int) (e.g., 2-norm, sup-norm)!")

    disturbance_norm = kwargs["disturbance_norm"]
    disturbance_norm_bound = kwargs["disturbance_norm_bound"]

    if disturbance_norm == 2:
        robustness_term = robustness_two_norm(disturbance_norm_bound)

    elif disturbance_norm == jnp.inf:
        robustness_term = robustness_sup_norm(disturbance_norm_bound)

    @jit
    def compute_clf_constraints(t: float, x: State) -> Tuple[Array, Array]:
        """Computes CBF and CLF constraints."""
        nonlocal a_clf, b_clf
        data = {}
        dyn_f, dyn_g = dyn_func(x)

        if n_lfs > 0:
            lf_x, lj_x, _, dlf_t, lc_x = compute_lyapunov_values(t, x)

            a_clf = a_clf.at[:, :n_con].set(jnp.matmul(lj_x, dyn_g))
            b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f) + robustness_term(lj_x) + lc_x)
            if relaxable:
                a_clf = a_clf.at[:, -n_lfs:].set(-lc_x)
                b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f))

            complete = lax.cond(jnp.all(lf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["lfs"] = lf_x
            data["complete"] = complete

        return a_clf, b_clf, data

    return compute_clf_constraints
