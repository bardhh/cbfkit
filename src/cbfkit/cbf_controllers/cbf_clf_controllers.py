import jax.numpy as jnp
from jax import Array
from numpy import array as arr, float32
from typing import Union, Dict, Any

from cbfkit.utils.user_types import (
    BarrierCollectionCallable,
    ControllerCallable,
    ControllerCallableReturns,
    DynamicsCallable,
    LyapunovCollectionCallable,
    State,
)
from cbfkit.utils.solvers import qp_solver
from .utils import block_diag_matrix, interleave_arrays, compute_enclosed_area


def cbf_clf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable = lambda: ([], [], [], []),
    lyapunovs: LyapunovCollectionCallable = lambda: ([], [], [], [], []),
    control_limits: Array = jnp.array([100.0, 100.0]),
    alpha: Array = jnp.array([1.0]),
    R: Union[Array, None] = None,
) -> ControllerCallable:
    """
    Compute the solution to a control barrier function/control Lyapunov function based quadratic program seeking to minimize the deviation
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
    barrier_functions, barrier_jacobians, _, barrier_partials = barriers()
    lyapunov_functions, lyapunov_jacobians, _, lyapunov_partials, lyapunov_conditions = lyapunovs()

    if R is None:
        Rmat = jnp.eye(len(control_limits), float)
    else:
        Rmat = R

    M = len(control_limits)
    L = len(barrier_functions)
    K = len(lyapunov_functions)

    if len(alpha) != L:
        alpha = jnp.array(L * alpha.min())

    def controller(t: float, x: State) -> ControllerCallableReturns:
        dynamics_f, dynamics_g, _ = dynamics_func(x)

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
        if L > 0:
            bf_x = jnp.stack([bf(t, x) for bf in barrier_functions])
            bj_x = jnp.stack([bj(t, x) for bj in barrier_jacobians])
            dbf_t = jnp.stack([bt(t, x) for bt in barrier_partials])
            Acbf = Acbf.at[:, :].set(-jnp.matmul(bj_x, dynamics_g))
            bcbf = bcbf.at[:].set(
                dbf_t + jnp.matmul(bj_x, dynamics_f) + jnp.multiply(jnp.array(alpha), bf_x)
            )

            # # Suppress angular control
            # Acbf = Acbf.at[:, 1].set(0)

        # Formulate CLF constraint(s)
        Aclf = jnp.zeros((K, M))
        bclf = jnp.zeros((K,))
        lf_x = jnp.zeros((K,))
        if K > 0:
            lf_x = jnp.stack([lf(t, x) for lf in lyapunov_functions])
            lj_x = jnp.stack([lj(t, x) for lj in lyapunov_jacobians])
            dlf_t = jnp.stack([lt(t, x) for lt in lyapunov_partials])
            lc_x = jnp.stack(
                [lc(lf(t, x)) for lc, lf in zip(lyapunov_conditions, lyapunov_functions)]
            )

            Aclf = Aclf.at[:, :].set(jnp.matmul(lj_x, dynamics_g))
            bclf = bclf.at[:].set(-dlf_t - jnp.matmul(lj_x, dynamics_f) + lc_x)

        # Formulate complete set of inequality constraints
        A = arr(jnp.vstack([Au, Acbf, Aclf]), dtype=float)
        b = arr(jnp.hstack([bu, bcbf, bclf]), dtype=float)
        feasible_area = compute_enclosed_area(A, b, control_limits)

        # Solve the QP
        sol, status = qp_solver(H, f, A, b)

        if not status:
            u = jnp.array([(-2 * x[2] * x[3] - x[3] ** 2 - 1 / 2 * x[2] ** 2) / x[2], -5 * x[6]])
        else:
            u = jnp.squeeze(jnp.array(sol[:M]))

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            "cbfs": bf_x,
            "clfs": lf_x,
            "feasible_area": feasible_area,
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
        }

        return u, data

    return controller


def adaptive_cbf_clf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable,
    lyapunovs: LyapunovCollectionCallable,
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
    barrier_functions, barrier_jacobians, _, barrier_partials = barriers()
    lyapunov_functions, lyapunov_jacobians, _, lyapunov_partials, lyapunov_conditions = lyapunovs()

    M = len(control_limits)
    L = len(barrier_functions)
    K = len(lyapunov_functions)

    if R is None:
        u_weights = jnp.array(M * [1])
        a_weights = jnp.array(L * [1e3])
        d_weights = jnp.array(K * [1e2])
        Rmat = jnp.diag(jnp.hstack([u_weights, a_weights, d_weights]))
    else:
        Rmat = R

    if len(alpha) != L:
        a_nom = jnp.array(L * alpha.min())
    else:
        a_nom = jnp.array(alpha)

    def controller(t: float, x: State) -> ControllerCallableReturns:
        dynamics_f, dynamics_g, _ = dynamics_func(x)

        u_nom, _ = nominal_input(t, x)
        H = arr(Rmat, dtype=float)
        f = arr(-2 * Rmat @ jnp.hstack([u_nom, a_nom, jnp.zeros((K,))]), dtype=float)

        # Formulate the input constraints
        alpha_limit = 100
        delta_limit = 1000
        upper_limits = jnp.hstack(
            [control_limits, jnp.array(L * [alpha_limit]), jnp.array(K * [delta_limit])]
        )
        lower_limits = jnp.hstack([control_limits, jnp.array(L * [0]), jnp.array(K * [0])])
        Au = block_diag_matrix(M + L + K)
        bu = interleave_arrays(upper_limits, lower_limits)

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M + L + K))
        bcbf = jnp.zeros((L,))
        bf_x = jnp.zeros((L,))
        if L > 0:
            bf_x = jnp.stack([bf(t, x) for bf in barrier_functions])
            bj_x = jnp.stack([bj(t, x) for bj in barrier_jacobians])
            dbf_t = jnp.stack([bt(t, x) for bt in barrier_partials])
            Acbf = Acbf.at[:, :M].set(-jnp.matmul(bj_x, dynamics_g))
            Acbf = Acbf.at[:, M : M + L].set(-jnp.diag(bf_x))
            bcbf = bcbf.at[:].set(dbf_t + jnp.matmul(bj_x, dynamics_f))

            # # Suppress angular control
            # Acbf = Acbf.at[:, 1].set(0)

        # Formulate CLF constraint(s)
        Aclf = jnp.zeros((K, M + L + K))
        bclf = jnp.zeros((K,))
        lf_x = jnp.zeros((K,))
        if K > 0:
            lf_x = jnp.stack([lf(t, x) for lf in lyapunov_functions])
            lj_x = jnp.stack([lj(t, x) for lj in lyapunov_jacobians])
            dlf_t = jnp.stack([lt(t, x) for lt in lyapunov_partials])
            lc_x = jnp.stack(
                [lc(lf(t, x)) for lc, lf in zip(lyapunov_conditions, lyapunov_functions)]
            )

            Aclf = Aclf.at[:, :M].set(jnp.matmul(lj_x, dynamics_g))
            Aclf = Aclf.at[:, M + L :].set(-jnp.ones((K,)))
            bclf = bclf.at[:].set(-dlf_t - jnp.matmul(lj_x, dynamics_f) + lc_x)

        # Formulate complete set of inequality constraints
        A = arr(jnp.vstack([Au, Acbf, Aclf]), dtype=float)
        b = arr(jnp.hstack([bu, bcbf, bclf]), dtype=float)

        # Solve the QP
        sol, status = qp_solver(H, f, A, b)
        u = jnp.squeeze(jnp.array(sol[:M]))

        # Saturate the solution if necessary
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            "cbfs": bf_x,
            "clfs": lf_x,
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
        }

        return u, data

    return controller
