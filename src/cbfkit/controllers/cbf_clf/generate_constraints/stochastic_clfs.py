"""
#! docstring
"""

from typing import Dict, Any, Tuple
from jax import Array, jit, lax
import jax.numpy as jnp
from cbfkit.utils.user_types import (
    DynamicsCallable,
    CertificateCollection,
    State,
)
from .unpack import unpack_for_clf
from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)


#! To Do: implement (after discovering theory)
####################################################################################################
### STOCHASTIC CLF: TBD #########################################################
def generate_compute_stochastic_clf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = ([], [], [], [], []),
    lyapunovs: CertificateCollection = ([], [], [], [], []),
    **kwargs: Dict[str, Any],
):
    """Placeholder. Theory still in development."""
    """
    #! To Do: docstring
    """
    compute_lyapunov_values = generate_compute_certificate_values(lyapunovs)
    n_con, _n_bfs, n_lfs, a_clf, b_clf, relaxable = unpack_for_clf(
        control_limits, lyapunovs, barriers, **kwargs
    )

    # Check for sigma function
    if "sigma" not in kwargs:
        raise ValueError("kwargs missing ra_params object!")
    sigma = kwargs["sigma"]
    if not callable(sigma):
        raise ValueError("sigma must be of type Callable[[Array], Array]!")

    @jit
    def compute_clf_constraints(t: float, x: State) -> Tuple[Array, Array]:
        """Computes CBF and CLF constraints."""
        nonlocal a_clf, b_clf
        data = {}
        dyn_f, dyn_g = dyn_func(x)
        s = sigma(x)

        if n_lfs > 0:
            lf_x, lj_x, lh_x, dlf_t, lc_x = compute_lyapunov_values(t, x)
            traces = jnp.array(
                [0.5 * jnp.trace(jnp.matmul(jnp.matmul(s.T, lh_ii), s)) for lh_ii in lh_x]
            )

            a_clf = a_clf.at[:, :n_con].set(jnp.matmul(lj_x, dyn_g))
            b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f) - traces + lc_x)
            if relaxable:
                a_clf = a_clf.at[:, -n_lfs:].set(-lc_x)
                b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f))

            complete = lax.cond(jnp.all(lf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["lfs"] = lf_x
            data["complete"] = complete

        return a_clf, b_clf, data

    return compute_clf_constraints
