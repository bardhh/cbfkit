"""Consolidated CBF constraints: meta-barrier combining multiple barriers."""

from typing import Any, Callable, List, Tuple, cast

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit

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

from ._constraint_core import build_cbf_constraint_generator


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
    """Jacobian for the consolidated constraint function."""
    jacobian = jacfwd(ccbf(constraint_functions, **kwargs))

    @jit
    def func(state_and_time: Array) -> Array:
        return jacobian(state_and_time)

    return func


def ccbf_hess(
    constraint_functions: List[CertificateCallable],
    **kwargs,
) -> Callable[[Array], Array]:
    """Hessian for the consolidated constraint function."""
    hessian = jacrev(jacfwd(ccbf(constraint_functions, **kwargs)))

    @jit
    def func(state_and_time: Array) -> Array:
        return hessian(state_and_time)

    return func


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

    return build_cbf_constraint_generator(
        control_limits,
        dyn_func,
        barriers,
        lyapunovs,
        certificate_package=consolidated_barriers,
        **kwargs,
    )
