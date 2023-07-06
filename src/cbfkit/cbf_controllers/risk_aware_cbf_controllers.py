from typing import Callable

import jax.numpy as jnp
from jax import Array
from numpy import array as arr
from scipy.special import erfinv
from typing import List, Union
from cbfkit.utils.user_types import (
    BarrierCollectionCallable,
    ControllerCallable,
    ControllerCallableReturns,
    DynamicsCallable,
    LyapunovCollectionCallable,
    State,
)
from cbfkit.utils.numerical_integration import forward_euler as integrator
from cbfkit.utils.solvers import qp_solver
from .utils import (
    block_diag_matrix,
    interleave_arrays,
    stochastic_barrier_transform,
    stochastic_hessian_transform,
    stochastic_jacobian_transform,
)

global INTEGRATOR_STATES, RB, RV, ETAB, ETAV


def risk_aware_cbf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable = lambda: ([], [], [], []),
    control_limits: Array = jnp.array([1000.0, 1000.0]),
    alpha: Array = jnp.array([1.0]),
    R: Union[Array, None] = None,
) -> ControllerCallable:
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
    global INTEGRATOR_STATES
    barrier_functions, barrier_jacobians, barrier_hessians, barrier_partials = barriers()

    if R is None:
        Rmat = jnp.eye(len(control_limits), dtype=float)
    else:
        Rmat = R

    M = len(control_limits)
    L = len(barrier_functions)
    INTEGRATOR_STATES = jnp.zeros((L,))

    if len(alpha) != L:
        alpha = jnp.array(L * alpha.min())

    #! TO DO -- Implement Programmatically
    eps = 0.01
    T = 10.0
    rhob = 0.1
    pv = 0.99

    def controller(t: float, x: State) -> ControllerCallableReturns:
        global INTEGRATOR_STATES, RB, RV, ETAB, ETAV
        dynamics_f, dynamics_g, dynamics_s = dynamics_func(x)
        N = len(dynamics_f)
        if t == 0:
            INTEGRATOR_STATES = jnp.zeros((L,))
            RB = stochastic_barrier_transform(jnp.array([bf(t, x) for bf in barrier_functions]))
            ETAB = jnp.linalg.norm(jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0.0]) @ dynamics_s)

        u_nom, _ = nominal_input(t, x)
        H = arr(Rmat, dtype=float)
        f = arr(-2 * Rmat @ u_nom, dtype=float)

        # Formulate the input constraints
        Au = block_diag_matrix(M)
        bu = interleave_arrays(control_limits, control_limits)

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M))
        bcbf = jnp.zeros((L,))
        bf_x = jnp.zeros((L,))
        bj_x = jnp.zeros((L, N))
        bh_x = jnp.zeros((L, N, N))
        if L > 0:
            bf_x = jnp.stack([bf(t, x) for bf in barrier_functions])
            bj_x = jnp.stack(
                [
                    stochastic_jacobian_transform(bf_x[ii], bj(t, x))
                    for ii, bj in enumerate(barrier_jacobians)
                ]
            )
            bh_x = jnp.stack(
                [
                    stochastic_hessian_transform(bf_x[ii], bj(t, x), bh(t, x))
                    for ii, (bj, bh) in enumerate(zip(barrier_jacobians, barrier_hessians))
                ]
            )
            dbf_t = jnp.stack(
                [
                    stochastic_jacobian_transform(bf_x[ii], bt(t, x))
                    for ii, bt in enumerate(barrier_partials)
                ]
            )
            h_vals = 1 - INTEGRATOR_STATES[:L] - RB - (ETAB * jnp.sqrt(2 * T)) * erfinv(1 - rhob)
            traces = jnp.array(
                [0.5 * jnp.trace(dynamics_s.T @ bh_ii @ dynamics_s) for bh_ii in bh_x]
            )

            Acbf = Acbf.at[:, :].set(jnp.matmul(bj_x, dynamics_g))
            bcbf = bcbf.at[:].set(
                jnp.multiply(jnp.array(alpha), h_vals)
                - dbf_t
                - jnp.matmul(bj_x, dynamics_f)
                - traces
            )

        # Formulate complete set of inequality constraints
        A = arr(jnp.vstack([Au, Acbf]), dtype=float)
        b = arr(jnp.hstack([bu, bcbf]), dtype=float)

        # Solve the QP
        sol, status = qp_solver(H, f, A, b)

        if not status:
            raise ValueError("INFEASIBLE QP")

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            "cbfs": bf_x,
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
        }

        # Update integrator state (discrete-implementation variant)
        jacobians1 = jnp.vstack([bj_x])
        hessians1 = jnp.vstack([bh_x])
        generators1 = jnp.matmul(
            jacobians1, dynamics_f + jnp.matmul(dynamics_g, u)
        ) + 0.5 * jnp.trace(
            jnp.matmul(jnp.matmul(dynamics_s.T, hessians1), dynamics_s), axis1=1, axis2=2
        )
        x2 = integrator(x, (dynamics_f + jnp.matmul(dynamics_g, u)) * 0.95, dt=0.01)
        dynamics_f2, dynamics_g2, dynamics_s2 = dynamics_func(x2)

        bf_x2 = jnp.zeros((L,))
        bj_x2 = jnp.zeros((L, N))
        bh_x2 = jnp.zeros((L, N, N))
        if L > 0:
            bf_x2 = jnp.stack([bf(t, x2) for bf in barrier_functions])
            bj_x2 = jnp.stack(
                [
                    stochastic_jacobian_transform(bf_x2[ii], bj(t, x2))
                    for ii, bj in enumerate(barrier_jacobians)
                ]
            )
            bh_x2 = jnp.stack(
                [
                    stochastic_hessian_transform(bf_x2[ii], bj(t, x2), bh(t, x2))
                    for ii, (bj, bh) in enumerate(zip(barrier_jacobians, barrier_hessians))
                ]
            )

        jacobians2 = jnp.vstack([bj_x2])
        hessians2 = jnp.vstack([bh_x2])
        generators2 = jnp.matmul(
            jacobians2, dynamics_f2 + jnp.matmul(dynamics_g2, u)
        ) + 0.5 * jnp.trace(
            jnp.matmul(jnp.matmul(dynamics_s2.T, hessians2), dynamics_s2), axis1=1, axis2=2
        )
        INTEGRATOR_STATES = integrator(INTEGRATOR_STATES, (generators1 + generators2) / 2, dt=0.01)

        return u, data

    return controller


def adaptive_risk_aware_cbf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable = lambda: ([], [], [], []),
    control_limits: Array = jnp.array([100.0, 100.0]),
    alpha: Array = jnp.array([1.0]),
    R: Union[Array, None] = None,
) -> ControllerCallable:
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
    global INTEGRATOR_STATES
    barrier_functions, barrier_jacobians, barrier_hessians, barrier_partials = barriers()

    M = len(control_limits)
    L = len(barrier_functions)

    if R is None:
        u_weights = jnp.array(M * [1])
        a_weights = jnp.array(L * [1000])
        Rmat = jnp.diag(jnp.hstack([u_weights, a_weights], dtype=float))
    else:
        Rmat = R

    INTEGRATOR_STATES = jnp.zeros((L,))

    if len(alpha) != L:
        a_nom = jnp.array(L * [alpha.min()])
    else:
        a_nom = jnp.array(alpha)

    #! TO DO -- Implement Programmatically
    T = 5.0
    rhob = 0.5

    def controller(t: float, x: State) -> ControllerCallableReturns:
        global INTEGRATOR_STATES, RB, RV, ETAB, ETAV
        dynamics_f, dynamics_g, dynamics_s = dynamics_func(x)
        N = len(dynamics_f)

        if t == 0:
            RB = stochastic_barrier_transform(jnp.array([bf(t, x) for bf in barrier_functions]))
            ETAB = jnp.linalg.norm(jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0.0]) @ dynamics_s)

        u_nom, _ = nominal_input(t, x)
        H = arr(Rmat, dtype=float)
        f = arr(-2 * Rmat @ jnp.hstack([u_nom, a_nom]), dtype=float)

        # Formulate the input constraints
        alpha_limit = 10
        upper_limits = jnp.hstack([control_limits, jnp.array(L * [alpha_limit])])
        lower_limits = jnp.hstack([control_limits, jnp.array(L * [0])])
        Au = block_diag_matrix(M + L + K)
        bu = interleave_arrays(upper_limits, lower_limits)

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M + L))
        bcbf = jnp.zeros((L,))
        bf_x = jnp.zeros((L,))
        bj_x = jnp.zeros((L, N))
        bh_x = jnp.zeros((L, N, N))
        if L > 0:
            bf_x = jnp.stack([bf(t, x) for bf in barrier_functions])
            bj_x = jnp.stack(
                [
                    stochastic_jacobian_transform(bf_x[ii], bj(t, x))
                    for ii, bj in enumerate(barrier_jacobians)
                ]
            )
            bh_x = jnp.stack(
                [
                    stochastic_hessian_transform(bf_x[ii], bj(t, x), bh(t, x))
                    for ii, (bj, bh) in enumerate(zip(barrier_jacobians, barrier_hessians))
                ]
            )
            dbf_t = jnp.stack(
                [
                    stochastic_jacobian_transform(bf_x[ii], bt(t, x))
                    for ii, bt in enumerate(barrier_partials)
                ]
            )
            h_vals = 1 - INTEGRATOR_STATES[:L] - RB - (ETAB * jnp.sqrt(2 * T)) * erfinv(1 - rhob)
            traces = jnp.array(
                [0.5 * jnp.trace(dynamics_s.T @ bh_ii @ dynamics_s) for bh_ii in bh_x]
            )

            Acbf = Acbf.at[:, :M].set(jnp.matmul(bj_x, dynamics_g))
            Acbf = Acbf.at[:, M : M + L].set(-jnp.diag(h_vals))
            bcbf = bcbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dynamics_f) - traces)

        # Formulate complete set of inequality constraints
        A = arr(jnp.vstack([Au, Acbf]), dtype=float)
        b = arr(jnp.hstack([bu, bcbf]), dtype=float)

        # Solve the QP
        sol, status = qp_solver(H, f, A, b)

        if not status:
            raise ValueError("INFEASIBLE QP")

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            "cbfs": bf_x,
            "clfs": lf_x,
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
        }

        jacobians = jnp.vstack([bj_x])
        hessians = jnp.vstack([bh_x])
        generators = jnp.matmul(
            jacobians, dynamics_f + jnp.matmul(dynamics_g, u)
        ) + 0.5 * jnp.trace(
            jnp.matmul(jnp.matmul(dynamics_s.T, hessians), dynamics_s), axis1=1, axis2=2
        )
        INTEGRATOR_STATES = integrator(INTEGRATOR_STATES, generators, dt=0.01)

        return u, data

    return controller
