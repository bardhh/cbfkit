"""
#! To Do: docstring
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
from .unpack import unpack_for_cbf


###################################################################################################
### RISK-AWARE PATH-INTEGRAL CBF: LfB + LgB*u + 0.5*Tr[sigma.T * d2B/dx2 * sigma] <= alpha(h) #####
def generate_compute_ra_pi_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Dict[str, Any],
) -> Callable[[Time, State], Tuple[Array, Array, Dict[str, Any]]]:
    """
    #! To Do: docstring
    """
    conditions = barriers[-1]
    compute_barrier_values = generate_compute_certificate_values(barriers)
    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable, relaxable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    # Check for Risk-Aware Params object
    if "ra_cbf_params" in kwargs:
        ra_params: RiskAwareParams = kwargs["ra_cbf_params"]  # type: ignore[assignment]
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
            gamma=jnp.zeros(n_bfs),  # Initialize gamma
            integrator_states=jnp.zeros((n_bfs,)),  # Initialize integrator_states
        )
        r_buffer = 0.0

    ra_params.integrator_states = jnp.zeros((n_bfs,))

    @jit
    def compute_cbf_constraints(t: Time, x: State) -> Tuple[Array, Array, Dict[str, Any]]:
        """Computes CBF and CLF constraints."""
        nonlocal a_cbf, b_cbf, ra_params
        data: Dict[str, Any] = {}
        dyn_f, dyn_g = dyn_func(x)
        assert ra_params.sigma is not None
        sigma = ra_params.sigma(x)
        ra_params.integrator_states = lax.cond(
            t == 0, lambda _: jnp.zeros((n_bfs,)), lambda _: ra_params.integrator_states, 0
        )

        if n_bfs > 0:
            bf_x, bj_x, bh_x, dbf_t, _ = compute_barrier_values(t, x)
            assert ra_params.gamma is not None
            # Ensure array types for addition
            w_vals = ra_params.integrator_states + ra_params.gamma + r_buffer
            bc_x = jnp.stack([bc(w_vals[ii]) for ii, bc in enumerate(conditions)])
            traces = jnp.array(
                [0.5 * jnp.trace(jnp.matmul(jnp.matmul(sigma.T, bh_ii), sigma)) for bh_ii in bh_x]
            )

            # Configure constraint matrix and vector (a * u <= b)
            a_cbf = a_cbf.at[:, :n_con].set(jnp.matmul(bj_x, dyn_g))
            b_cbf = b_cbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dyn_f) - traces + bc_x)
            if tunable:
                a_cbf = a_cbf.at[:, n_con:n_bfs].set(-bc_x)
                b_cbf = b_cbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dyn_f) - traces)

            violated = lax.cond(jnp.any(bf_x > 1), lambda _fake: True, lambda _fake: False, 0)

            data["bfs"] = bf_x
            data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints
