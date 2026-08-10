"""
risk_aware_path_integral_cbfs.py
=================================

.. deprecated::
    **NON-FUNCTIONAL STUB — the path integral never accumulates.**

    The generator body unconditionally overwrites ``ra_params.integrator_states``
    with zeros, discarding any caller-supplied value::

        ra_params.integrator_states = jnp.zeros((n_bfs,))

    No term anywhere in the package ever increments it, so the reset-on-``t == 0``
    ``lax.cond`` reduces to ``w <- (t == 0 ? 0 : w)`` over a value that is already
    zero: ``integrator_states`` is identically 0 at every timestep.  The root cause
    is architectural: this function receives only ``(t, x)`` and cannot accumulate
    mutable state across calls inside a JIT/scan loop without a proper carry
    mechanism.

    **Use** ``accumulating_risk_aware_cbf_controller`` from
    ``cbfkit.controllers.cbf_clf.accumulating_risk_aware_cbf`` instead.  It carries
    ``I_L`` correctly in ``ControllerData.sub_data["I_L"]`` and is JIT/scan-safe.

    This module is kept for backward compatibility only.  Its runtime behaviour is
    unchanged so that existing tests continue to pass.
"""

from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax

from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CbfClfQpData,
    CertificateCollection,
    DynamicsCallable,
    State,
    Time,
)

from ._constraint_core import batched_hessian_trace
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
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    conditions = barriers[-1]
    compute_barrier_values = generate_compute_certificate_values(barriers)
    n_con, n_bfs, _n_lfs, a_cbf_template, b_cbf_template, tunable, relaxable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    # Check for Risk-Aware Params object
    if "ra_cbf_params" in kwargs:
        ra_params: RiskAwareParams = kwargs["ra_cbf_params"]  # type: ignore[assignment]
    else:
        ra_params = RiskAwareParams(
            sigma=lambda x: jnp.zeros((x.shape[0], 1)),
            gamma=jnp.zeros(n_bfs),  # Initialize gamma
            integrator_states=jnp.zeros((n_bfs,)),  # Initialize integrator_states
        )

    ra_params.integrator_states = jnp.zeros((n_bfs,))
    integrator_states_template = ra_params.integrator_states

    @jit
    def compute_cbf_constraints(t: Time, x: State) -> Tuple[Array, Array, CbfClfQpData]:
        """Computes CBF and CLF constraints."""
        data: CbfClfQpData = {}
        dyn_f, dyn_g = dyn_func(x)
        assert ra_params.sigma is not None
        sigma = ra_params.sigma(x)

        # Bind the zero templates to locals. Rebinding the enclosing names instead
        # would store this trace's tracers in the closure, so the next trace (new
        # dtype/shape, or disable_jit) would read a leaked tracer.
        a_cbf = a_cbf_template
        b_cbf = b_cbf_template

        # Per the module docstring this integrator never accumulates: no term
        # anywhere increments it, so both branches yield the construction-time
        # zeros and the reset is a no-op. It is kept (reading the concrete
        # template, never the mutated attribute) so the reset-at-t==0 intent
        # survives for whoever threads a real carry through ControllerData.
        integrator_states = lax.cond(
            t == 0, lambda _: jnp.zeros((n_bfs,)), lambda _: integrator_states_template, 0
        )

        if n_bfs > 0:
            bf_x, bj_x, bh_x, dbf_t, _ = compute_barrier_values(t, x)
            # Ensure array types for addition
            w_vals = integrator_states
            bc_x = jnp.stack([bc(w_vals[ii]) for ii, bc in enumerate(conditions)])
            traces = batched_hessian_trace(sigma, bh_x)

            # Configure constraint matrix and vector (a * u <= b)
            a_cbf = a_cbf.at[:, :n_con].set(jnp.matmul(bj_x, dyn_g))
            b_cbf = b_cbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dyn_f) - traces + bc_x)
            if tunable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-bc_x)
                b_cbf = b_cbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dyn_f) - traces)

            violated = lax.cond(jnp.any(bf_x > 1), lambda _fake: True, lambda _fake: False, 0)

            data["bfs"] = bf_x
            data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints
