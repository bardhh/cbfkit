from typing import Callable

import jax.numpy as jnp
from jax import grad, jit
from jax.interpreters.xla import DeviceArray
from kvxopt import matrix, solvers
from numpy import array as arr
from scipy.linalg import block_diag
from scipy.special import erfinv

from ..solvers import qp_solver
from .utils import (
    block_diag_matrix,
    interleave_arrays,
    stochastic_barrier_transform,
    stochastic_hessian_transform,
    stochastic_jacobian_transform,
)


def stochastic_cbf_controller(
    nominal_input: Callable,
    dynamics_func: Callable,
    barrier_funcs: Callable,
    barrier_jacobians: Callable,
    barrier_hessians: Callable,
    control_limits: DeviceArray = jnp.array([100.0, 100.0]),
    alpha: float = 1,
    R: DeviceArray = None,
) -> Callable[DeviceArray, DeviceArray]:
    """
    Compute the solution to a control barrier function based quadratic program seeking to minimize the deviation
    from some nominal input.

    Args:
    nominal_input: the function in charge of computing the nominal input
    dynamics_func: the dynamics function describing the system to be controlled.
    barrier_func: the barrier function which should be less than or equal to zero within a particular set.
    barrier_jacobian: the Jacobian of the barrier function.
    control_limits: denotes the maximum control input in magnitude (e.g., for a car it's acceleration and steering rate).
    Q: state-weight parameter in the QP.
    R: input-weight parameter in the QP.
    P: positive definite matrix to compute final cost

    Returns:
    u: Optimal control input
    """
    if R is None:
        R = jnp.eye(len(control_limits), dtype=float)

    M = len(control_limits)
    L = len(barrier_funcs)

    #! TO DO -- Implement Programmatically
    beta = 0.1

    def controller(x):
        dynamics = dynamics_func(x)
        N = len(dynamics[0])
        if len(dynamics) == 2:
            # Deterministic Model
            dynamics_f, dynamics_g = dynamics
            dynamics_s = jnp.zeros((N, N))
        elif len(dynamics) == 3:
            # Stochastic Model
            dynamics_f, dynamics_g, dynamics_s = dynamics

        H = matrix(arr(R, dtype=float))
        f = matrix(arr(-2 * R @ nominal_input(x), dtype=float))

        # Formulate the input constraints
        Au = block_diag_matrix(M)
        bu = interleave_arrays(control_limits, control_limits)

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M))
        bcbf = jnp.zeros((L,))
        trace_term = jnp.zeros((L,))
        for ib, (bf, bj, bh) in enumerate(zip(barrier_funcs, barrier_jacobians, barrier_hessians)):
            bf_val, bj_val, bh_val = bf(x), bj(x), bh(x)
            func = stochastic_barrier_transform(bf_val)
            jaco = stochastic_jacobian_transform(bf_val, bj_val)
            hess = stochastic_hessian_transform(bf_val, bj_val, bh_val)
            trace_term = trace_term.at[ib].set(0.5 * jnp.trace(dynamics_s.T @ hess @ dynamics_s))
            Acbf = Acbf.at[ib, :].set(jaco @ dynamics_g)
            bcbf = bcbf.at[ib].set(beta - alpha * func - jaco @ dynamics_f - trace_term[ib])

        # Formulate complete set of inequality constraints
        A = matrix(arr(jnp.vstack([Au, Acbf]), dtype=float))
        b = matrix(arr(jnp.hstack([bu, bcbf]), dtype=float))

        # Solve the QP
        sol = qp_solver(H, f, A, b)

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        return u

    return controller


def adaptive_stochastic_cbf_controller(
    nominal_input: Callable,
    dynamics_func: Callable,
    barrier_funcs: Callable,
    barrier_jacobians: Callable,
    barrier_hessians: Callable,
    control_limits: DeviceArray = jnp.array([100.0, 100.0]),
    alpha: float = 1,
    R: DeviceArray = None,
) -> Callable[DeviceArray, DeviceArray]:
    """
    Compute the solution to a control barrier function based quadratic program seeking to minimize the deviation
    from some nominal input.

    Args:
    nominal_input: the function in charge of computing the nominal input
    dynamics_func: the dynamics function describing the system to be controlled.
    barrier_func: the barrier function which should be less than or equal to zero within a particular set.
    barrier_jacobian: the Jacobian of the barrier function.
    control_limits: denotes the maximum control input in magnitude (e.g., for a car it's acceleration and steering rate).
    Q: state-weight parameter in the QP.
    R: input-weight parameter in the QP.
    P: positive definite matrix to compute final cost

    Returns:
    u: Optimal control input
    """
    M = len(control_limits)
    L = len(barrier_funcs)

    if R is None:
        u_weights = jnp.array(M * [1])
        a_weights = jnp.array(L * [100])
        R = jnp.diag(jnp.hstack([u_weights, a_weights]))

    #! TO DO -- Implement Programmatically
    beta = 0.1

    def controller(x):
        dynamics = dynamics_func(x)
        N = len(dynamics[0])
        if len(dynamics) == 2:
            # Deterministic Model
            dynamics_f, dynamics_g = dynamics
            dynamics_s = jnp.zeros((N, N))
        elif len(dynamics) == 3:
            # Stochastic Model
            dynamics_f, dynamics_g, dynamics_s = dynamics

        unom = nominal_input(x)
        anom = jnp.array(L * [alpha])
        H = matrix(arr(R, dtype=float))
        f = matrix(arr(-2 * R @ jnp.hstack([unom, anom]), dtype=float))

        # Formulate the input constraints
        alpha_limit = 100
        upper_limits = jnp.hstack([control_limits, jnp.array(L * [alpha_limit])])
        lower_limits = jnp.hstack([control_limits, jnp.array(L * [0])])
        Au = block_diag_matrix(M + L)
        bu = interleave_arrays(upper_limits, lower_limits)

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M + L))
        bcbf = jnp.zeros((L,))
        trace_term = jnp.zeros((L,))
        for ib, (bf, bj, bh) in enumerate(zip(barrier_funcs, barrier_jacobians, barrier_hessians)):
            bf_val, bj_val, bh_val = bf(x), bj(x), bh(x)
            func = stochastic_barrier_transform(bf_val)
            jaco = stochastic_jacobian_transform(bf_val, bj_val)
            hess = stochastic_hessian_transform(bf_val, bj_val, bh_val)
            trace_term = trace_term.at[ib].set(0.5 * jnp.trace(dynamics_s.T @ hess @ dynamics_s))
            Acbf = Acbf.at[ib, :M].set(jaco @ dynamics_g)
            Acbf = Acbf.at[ib, M + ib].set(-beta)
            bcbf = bcbf.at[ib].set(-alpha * func - jaco @ dynamics_f - trace_term[ib])

        # Formulate complete set of inequality constraints
        A = matrix(arr(jnp.vstack([Au, Acbf]), dtype=float))
        b = matrix(arr(jnp.hstack([bu, bcbf]), dtype=float))

        # Solve the QP
        sol = qp_solver(H, f, A, b)

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        return u

    return controller

