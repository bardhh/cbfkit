"""
#! docstring
"""

from typing import Tuple, Dict, Any
import jax.numpy as jnp
from jax import Array, jit, lax, scipy
from cbfkit.utils.user_types import (
    DynamicsCallable,
    CertificateCollection,
    State,
)
from .unpack import unpack_for_cbf
from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)


#! To Do: implement (after developing theory)
####################################################################################################
### RISK-AWARE CBF: TBD #########################################################
def generate_compute_ra_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = ([], [], [], [], []),
    lyapunovs: CertificateCollection = ([], [], [], [], []),
    **kwargs: Dict[str, Any],
):
    """Placeholder. Theory still in development."""
    compute_barrier_values = generate_compute_certificate_values(barriers)

    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    # Check for Risk-Aware Params object
    if "ra_params" not in kwargs:
        raise ValueError("kwargs missing ra_params object!")

    ra_params = kwargs["ra_params"]
    r_buffer = float(
        jnp.sqrt(2 * ra_params.ra_clf.t_max)
        * ra_params.ra_clf.eta
        * scipy.special.erfinv(2 * ra_params.ra_clf.p_bound - 1)
    )

    @jit
    def compute_cbf_constraints(t: float, x: State) -> Tuple[Array, Array]:
        """Computes CBF and CLF constraints."""
        nonlocal a_cbf, b_cbf
        data = {}
        dyn_f, dyn_g = dyn_func(x)
        sigma = ra_params.sigma(x)

        if n_bfs > 0:
            bf_x, bj_x, bh_x, dbf_t, bc_x = compute_barrier_values(t, x)
            traces = jnp.array(
                [0.5 * jnp.trace(jnp.matmul(jnp.matmul(sigma.T, bh_ii), sigma)) for bh_ii in bh_x]
            )

            # # Configure constraint matrix and vector (a * u <= b)
            # a_cbf = a_cbf.at[:, :n_con].set(jnp.matmul(bj_x, dyn_g))
            # b_cbf = b_cbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dyn_f) - traces + bc_x)
            # if tunable:
            #     a_cbf = a_cbf.at[:, n_con:n_bfs].set(-bc_x)
            #     b_cbf = b_cbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dyn_f) - traces)

            # violated = lax.cond(jnp.any(bf_x > 1), lambda _fake: True, lambda _fake: False, 0)

            # data["bfs"] = bf_x
            # data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints
