from typing import Any, Callable, Dict, Tuple

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
from .unpack import unpack_for_cbf


###################################################################################################
### STOCHASTIC CBF: LfB + LgB*u + 0.5*Tr[sigma.T * d2B/dx2 * sigma] >= -alpha*B + beta ############
def generate_compute_stochastic_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    compute_barrier_values = generate_compute_certificate_values(barriers)
    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable, relaxable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    # Check for sigma function
    if "sigma" not in kwargs:
        raise ValueError("kwargs missing ra_params object!")
    sigma = kwargs["sigma"]
    if not callable(sigma):
        raise ValueError("sigma must be of type Callable[[Array], Array]!")

    @jit
    def compute_cbf_constraints(t: Time, x: State) -> Tuple[Array, Array, CbfClfQpData]:
        """Computes CBF and CLF constraints."""
        nonlocal a_cbf, b_cbf
        data: CbfClfQpData = {}
        dyn_f, dyn_g = dyn_func(x)
        s = sigma(x)

        if n_bfs > 0:
            bf_x, bj_x, bh_x, dbf_t, bc_x = compute_barrier_values(t, x)
            traces = jnp.array(
                [0.5 * jnp.trace(jnp.matmul(jnp.matmul(s.T, bh_ii), s)) for bh_ii in bh_x]
            )

            # Configure constraint matrix and vector (a * u <= b)
            # Safety: LfB + LgB u + 0.5 Tr(...) >= -alpha(B) + beta
            # LgB u >= -LfB - 0.5 Tr(...) - alpha(B) + beta
            # -LgB u <= LfB + 0.5 Tr(...) + alpha(B) - beta
            # Note: bc_x contains -alpha(B) + beta usually? Or just the RHS condition.
            # If certificate condition is gamma(h) (class K), then we want h_dot >= -gamma(h).
            # So RHS is -gamma(h).
            # zeroing_cbfs uses: b = dbf_t + Lfh + bc_x.
            # If bc_x is gamma(h), this means h_dot + gamma(h) >= 0 ? No.
            # zeroing_cbfs logic: b = ... + bc_x.
            # If bc_x is the class K function value (e.g. alpha * h), then `b` is the upper bound for `-LgB u`.
            # -LgB u <= LfB + gamma(h).
            # So LgB u >= -LfB - gamma(h).
            # h_dot >= -gamma(h). Correct.

            # Here: b_cbf.set(-dbf_t - jnp.matmul(bj_x, dyn_f) - traces + bc_x)
            # If a_cbf is positive LgB:
            # LgB u <= -LfB - Tr + bc_x.
            # LfB + LgB u + Tr <= bc_x.
            # This implies h_dot <= bc_x.
            # If bc_x is negative (e.g. -alpha h), then h_dot <= -alpha h. This is STABILITY/Convergence to 0.
            # If this is for Safety (stay positive), it should be >=.

            # Given standard CBF is >=, I suspect `stochastic_cbfs.py` was using sublevel sets or had a bug.
            # I will switch it to match zeroing_cbfs (negative a_cbf) to assume superlevel sets.

            a_cbf = a_cbf.at[:, :n_con].set(-jnp.matmul(bj_x, dyn_g))
            b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f) + traces + bc_x)

            if tunable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-bc_x)
                b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f) + traces)
            elif relaxable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-1.0)

            # For superlevel-set CBFs (h >= 0 means safe), violation occurs when h < 0
            violated = lax.cond(jnp.any(bf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["bfs"] = bf_x
            data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints
