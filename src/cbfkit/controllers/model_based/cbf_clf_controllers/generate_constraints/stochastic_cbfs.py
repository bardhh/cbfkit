"""
#! docstring
"""

from typing import Tuple, Dict, Any, Callable
import jax.numpy as jnp
from jax import Array, jit, lax
from cbfkit.utils.user_types import (
    DynamicsCallable,
    CertificateCollection,
    State,
)
from .unpack import unpack_for_cbf
from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)


###################################################################################################
### STOCHASTIC CBF: LfB + LgB*u + 0.5*Tr[sigma.T * d2B/dx2 * sigma] <= -alpha*B + beta ############
def generate_compute_stochastic_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = ([], [], [], [], []),
    lyapunovs: CertificateCollection = ([], [], [], [], []),
    **kwargs: Dict[str, Any],
):
    """
    #! To Do: docstring
    """
    compute_barrier_values = generate_compute_certificate_values(barriers)
    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    # Check for sigma function
    if "sigma" not in kwargs:
        raise ValueError("kwargs missing ra_params object!")
    sigma = kwargs["sigma"]
    if not callable(sigma):
        raise ValueError("sigma must be of type Callable[[Array], Array]!")

    @jit
    def compute_cbf_constraints(t: float, x: State) -> Tuple[Array, Array]:
        """Computes CBF and CLF constraints."""
        nonlocal a_cbf, b_cbf
        data = {}
        dyn_f, dyn_g = dyn_func(x)
        s = sigma(x)

        if n_bfs > 0:
            bf_x, bj_x, bh_x, dbf_t, bc_x = compute_barrier_values(t, x)
            traces = jnp.array(
                [0.5 * jnp.trace(jnp.matmul(jnp.matmul(s.T, bh_ii), s)) for bh_ii in bh_x]
            )

            # Configure constraint matrix and vector (a * u <= b)
            a_cbf = a_cbf.at[:, :n_con].set(jnp.matmul(bj_x, dyn_g))
            b_cbf = b_cbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dyn_f) - traces + bc_x)
            if tunable:
                a_cbf = a_cbf.at[:, n_con:n_bfs].set(jnp.expand_dims(-bc_x, axis=-1))
                b_cbf = b_cbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dyn_f) - traces)

            violated = lax.cond(jnp.any(bf_x > 1), lambda _fake: True, lambda _fake: False, 0)

            data["bfs"] = bf_x
            data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints
