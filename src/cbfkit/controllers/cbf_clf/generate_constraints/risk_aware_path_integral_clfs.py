"""
#! docstring
"""

from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax, scipy

from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CertificateCollection,
    DynamicsCallable,
    State,
    Time,
)

from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)
from .unpack import unpack_for_clf


###################################################################################################
### RISK-AWARE PATH-INTEGRAL CLF: LfV + LgV*u + 0.5*Tr[sigma.T * d2V/dx2 * sigma] <= c(V) #########
def generate_compute_ra_pi_clf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, Dict[str, Any]]]:
    """
    #! To Do: docstring
    """
    conditions = lyapunovs[-1]
    compute_lyapunov_values = generate_compute_certificate_values(lyapunovs)
    n_con, _n_bfs, n_lfs, a_clf, b_clf, relaxable = unpack_for_clf(
        control_limits, lyapunovs, barriers, **kwargs
    )

    # Check for Risk-Aware Params object
    if "ra_clf_params" in kwargs:
        ra_params: RiskAwareParams = kwargs["ra_clf_params"]  # type: ignore[assignment]
        assert ra_params.eta is not None
        assert ra_params.t_max is not None
        assert ra_params.p_bound is not None
        assert ra_params.gamma is not None
        r_buffer = float(
            ra_params.eta * jnp.sqrt(2 * ra_params.t_max) * scipy.special.erfinv(ra_params.p_bound)
        )
    else:
        ra_params = RiskAwareParams(
            sigma=lambda x: jnp.zeros((x.shape[0], 1)),
            gamma=jnp.zeros(n_lfs),  # Initialize gamma to zeros
            integrator_states=jnp.zeros((n_lfs,)),  # Initialize integrator_states
        )
        r_buffer = 0.0

    ra_params.integrator_states = jnp.zeros((n_lfs,))

    @jit
    def compute_clf_constraints(t: Time, x: State) -> Tuple[Array, Array, Dict[str, Any]]:
        """Computes CBF and CLF constraints."""
        nonlocal a_clf, b_clf, ra_params
        data: Dict[str, Any] = {}
        dyn_f, dyn_g = dyn_func(x)
        assert ra_params.sigma is not None
        sigma = ra_params.sigma(x)
        ra_params.integrator_states = lax.cond(
            t == 0, lambda _: jnp.zeros((n_lfs,)), lambda _: ra_params.integrator_states, 0
        )

        if n_lfs > 0:
            lf_x, lj_x, lh_x, dlf_t, _ = compute_lyapunov_values(t, x)
            assert ra_params.gamma is not None
            # Ensure array types for addition
            w_vals = ra_params.integrator_states + ra_params.gamma + r_buffer
            lc_x = jnp.stack([lc(w_vals[ii]) for ii, lc in enumerate(conditions)])
            traces = jnp.array(
                [0.5 * jnp.trace(jnp.matmul(jnp.matmul(sigma.T, lh_ii), sigma)) for lh_ii in lh_x]
            )

            # Configure constraint matrix and vector (a * u <= b)
            a_clf = a_clf.at[:, :n_con].set(jnp.matmul(lj_x, dyn_g))
            b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f) - traces + lc_x)
            if relaxable:
                a_clf = a_clf.at[:, -n_lfs:].set(-jnp.ones((n_lfs,)))
                b_clf = b_clf.at[:].set(-dlf_t - jnp.matmul(lj_x, dyn_f) - traces + lc_x)

            # Check whether goal set reached
            complete = lax.cond(jnp.all(lf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["lfs"] = w_vals
            data["lfs_nom"] = lf_x
            data["complete"] = complete

        return a_clf, b_clf, data

    return compute_clf_constraints
