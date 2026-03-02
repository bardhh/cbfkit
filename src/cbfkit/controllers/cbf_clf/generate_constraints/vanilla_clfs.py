from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax

from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CbfClfQpData,
    CertificateCollection,
    DynamicsCallable,
    State,
    Time,
)

from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)
from .unpack import unpack_for_clf


####################################################################################################
### VANILLA CLF: LfV + LgV*u <= conditions #########################################################
def generate_compute_vanilla_clf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    compute_lyapunov_values = generate_compute_certificate_values(
        lyapunovs, compute_hessians=False
    )
    n_con, _n_bfs, n_lfs, a_clf, b_clf, relaxable = unpack_for_clf(
        control_limits, lyapunovs, barriers, **kwargs
    )
    scale_clf = kwargs.get("scale_clf", 1.0)

    @jit
    def compute_clf_constraints(
        t: Time,
        x: State,
        f: Optional[Array] = None,
        g: Optional[Array] = None,
    ) -> Tuple[Array, Array, CbfClfQpData]:
        """Computes CBF and CLF constraints."""
        nonlocal a_clf, b_clf
        data: CbfClfQpData = {}

        dyn_f = f
        dyn_g = g
        if dyn_f is None or dyn_g is None:
            dyn_f, dyn_g = dyn_func(x)

        if n_lfs > 0:
            lf_x, lj_x, _, dlf_t, lc_x = compute_lyapunov_values(t, x)

            a_clf = a_clf.at[:, :n_con].set(jnp.matmul(lj_x, dyn_g))
            b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f) + lc_x)
            if relaxable:
                # Use additive relaxation (-scale_clf) instead of multiplicative (-lc_x * scale_clf).
                # Multiplicative relaxation vanishes at V=0, causing loss of authority and ill-conditioning.
                a_clf = a_clf.at[:, -n_lfs:].set(-scale_clf * jnp.eye(n_lfs))
                # Keep b_clf as is (with lc_x), realizing additive relaxation: V_dot <= lc_x + delta

            # Treat goal reached when Lyapunov value is sufficiently small
            complete = lax.cond(
                jnp.all(lf_x <= kwargs.get("clf_complete_tol", 1e-3)),
                lambda _fake: True,
                lambda _fake: False,
                0,
            )

            data["lfs"] = lf_x
            data["complete"] = complete

        return a_clf, b_clf, data

    return compute_clf_constraints
