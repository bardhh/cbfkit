from typing import Any, Callable, Dict, List, Tuple, cast

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit, lax

from cbfkit.certificates import certificate_package
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CbfClfQpData,
    CertificateCallable,
    CertificateCollection,
    DynamicsCallable,
    State,
    Time,
)

from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)
from .unpack import unpack_for_cbf


def ccbf(
    constraint_functions: List[CertificateCallable],
    **kwargs,
) -> Callable[[Array], Array]:
    n_bfs = len(constraint_functions)

    @jit
    def func(state_and_weights_and_time: Array) -> Array:
        t = state_and_weights_and_time[-1]
        x = state_and_weights_and_time[: -1 - 2 * n_bfs]
        w = state_and_weights_and_time[-1 - 2 * n_bfs : -1 - n_bfs]
        return 1 - jnp.sum(
            jnp.array([jnp.exp(-ww * cf(t, x)) for ww, cf in zip(w, constraint_functions)])
        )

    return func


def ccbf_grad(
    constraint_functions: List[CertificateCallable],
    **kwargs,
) -> Callable[[Array], Array]:
    """Jacobian for the constraint function defined by cbf.

        Args:
    ually populate

        Returns
        -------
            ret (float): value of constraint function evaluated at time and state
    """
    jacobian = jacfwd(ccbf(constraint_functions, **kwargs))

    @jit
    def func(state_and_time: Array) -> Array:
        """

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns
        -------
            Array: cbf jacobian (gradient)
        """
        return jacobian(state_and_time)

    return func


def ccbf_hess(
    constraint_functions: List[CertificateCallable],
    **kwargs,
) -> Callable[[Array], Array]:
    """Hessian for the constraint function defined by cbf.

        Args:
    ually populate

        Returns
        -------
            ret (float): value of constraint function evaluated at time and state
    """
    hessian = jacrev(jacfwd(ccbf(constraint_functions, **kwargs)))

    @jit
    def func(state_and_time: Array) -> Array:
        """

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns
        -------
            Array: cbf hessian
        """
        return hessian(state_and_time)

    return func


####################################################################################################
### Consolidated CBF: Lfh + Lgh * u + Lwh * wdot + \alpha(h) >= 0 ####################################################
def generate_compute_consolidated_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
    **kwargs: Any,
) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
    if "alpha" not in kwargs:
        raise ValueError("Missing Class K function alpha from kwargs!")
    if "n_states" not in kwargs:
        raise ValueError("Missing n_states from kwargs!")
    bfs, _, _, _, _ = barriers
    consolidated_barrier_package = certificate_package(
        ccbf, ccbf_grad, ccbf_hess, cast(int, kwargs["n_states"]) + len(bfs)
    )
    consolidated_barriers = consolidated_barrier_package(
        certificate_conditions=linear_class_k(alpha=cast(float, kwargs["alpha"])),
        constraint_functions=bfs,
    )
    compute_barrier_values = generate_compute_certificate_values(consolidated_barriers)

    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable, relaxable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )
    scale_cbf = kwargs.get("scale_cbf", 1.0)

    @jit
    def compute_cbf_constraints(t: Time, x: State) -> Tuple[Array, Array, CbfClfQpData]:
        """Computes CBF and CLF constraints."""
        nonlocal a_cbf, b_cbf
        data: CbfClfQpData = {}
        dyn_f, dyn_g = dyn_func(x)

        if n_bfs > 0:
            bf_x, bj_x, _, dbf_t, bc_x = compute_barrier_values(t, x)

            # Check if consolidated constraint required
            a_cbf = a_cbf.at[:, :n_con].set(-jnp.matmul(bj_x, dyn_g))
            b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f) + bc_x)
            if tunable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-scale_cbf * jnp.diag(bc_x))
                b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f))
            elif relaxable:
                a_cbf = a_cbf.at[:, n_con : n_con + n_bfs].set(-scale_cbf * jnp.eye(n_bfs))

            violated = lax.cond(jnp.any(bf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["bfs"] = bf_x
            data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints
