from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax

from cbfkit.controllers.cbf_clf.utils.barrier_activation import compute_activation_weights
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


####################################################################################################
### ACTIVATED CBF ##################################################################################
def generate_compute_activated_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    """
    Generates compute function for Activated CBF constraints.
    Scales constraints by activation weights based on proximity.
    """
    compute_barrier_values = generate_compute_certificate_values(barriers)

    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable, relaxable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    # Extract activation parameters
    obstacle_positions = kwargs.get("obstacle_positions")
    k_closest = kwargs.get("k_closest", 3)
    activation_radius = kwargs.get("activation_radius", 2.0)
    activation_smoothness = kwargs.get("activation_smoothness", 5.0)

    if obstacle_positions is None:
        # Fallback or warning? Assuming it's provided if this generator is used.
        # We can't easily print in JIT gen phase, but we can assert.
        # Actually, this function is called at generation time (not JIT), so we can check.
        # But obstacle_positions might be JAX array.
        pass

    @jit
    def compute_cbf_constraints(t: Time, x: State) -> Tuple[Array, Array, CbfClfQpData]:
        """Computes CBF and CLF constraints with activation."""
        nonlocal a_cbf, b_cbf
        data: CbfClfQpData = {}
        dyn_f, dyn_g = dyn_func(x)

        if n_bfs > 0:
            bf_x, bj_x, _, dbf_t, bc_x = compute_barrier_values(t, x)

            # Standard CBF logic
            a_cbf = a_cbf.at[:, :n_con].set(-jnp.matmul(bj_x, dyn_g))
            b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f) + bc_x)

            if tunable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-bc_x)
                b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f))
            elif relaxable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-1.0)

            # --- Activation Logic ---
            if obstacle_positions is not None:
                weights = compute_activation_weights(
                    x,
                    obstacle_positions,
                    k=k_closest,
                    radius=activation_radius,
                    smoothness=activation_smoothness,
                )

                # Apply weights (scale entire constraint row)
                # w * (A u <= b)  <=>  (w A) u <= (w b)
                # Here a_cbf represents 'A' in G u <= h (qp standard)
                # or actually a_cbf is LHS matrix, b_cbf is RHS vector?
                # In zeroing_cbfs:
                # a_cbf = -Lg h
                # b_cbf = Lf h + alpha
                # Constraint is: Lf h + Lg h u + alpha >= 0  =>  -Lg h u <= Lf h + alpha
                # So a_cbf u <= b_cbf. Correct.

                # Scale a_cbf rows
                # a_cbf is (n_bfs, n_vars)
                # weights is (n_bfs,)
                a_cbf = a_cbf * weights[:, None]
                b_cbf = b_cbf * weights

                data["activation_weights"] = weights

            violated = lax.cond(jnp.any(bf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["bfs"] = bf_x
            data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints
