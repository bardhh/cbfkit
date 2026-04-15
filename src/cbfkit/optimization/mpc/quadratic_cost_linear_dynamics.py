from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.optimization.quadratic_program.solver_registry import get_solver
from cbfkit.utils.user_types.callables import QpSolverCallable

_DEFAULT_SOLVER = get_solver("jaxopt")


def generate_mpc_solver_quadratic_cost_linear_dynamics(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    Qn: Array,
    N: int,
    solver: Optional[QpSolverCallable] = None,
) -> Callable:
    n = Qn.shape[0]
    m = B.shape[1]
    mpc_to_qp = generate_mpc_to_qp(A, B, Q, R, Qn, N)
    solve_qp = solver if solver is not None else get_solver("jaxopt")

    @jit
    def solve(concatenated_x_xr: Array) -> Tuple[Array, Array]:
        """Solves Discrete-Time, LTI MPC problem.

        Args:
            concatenated_x_xr (Array): concatenation of current state vector (x) and reference path (xr)

        Returns
        -------
            Tuple(Array, Array): (x_opt, u_opt) optimal state and input sequence
        """
        # Convert Discrete-Time, LTI MPC problem into QP and solve
        h_mat, f_vec, g_mat, h_vec, a_mat, b_vec = mpc_to_qp(concatenated_x_xr)
        qp_result = solve_qp(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)
        sol = qp_result.primal

        # if not status:
        #     raise ValueError("Infeasible MPC!")

        # Extract optimal state and control trajectories
        sol = jnp.array(sol).flatten()
        x_opt = sol[: (N + 1) * n].reshape((N + 1, n)).T
        u_opt = sol[(N + 1) * n :].reshape((N, m)).T

        return x_opt, u_opt

    return solve


def generate_mpc_to_qp(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    QN: Array,
    horizon: int,
) -> Callable:
    n_states, n_inputs = B.shape

    # Formulating the quadratic cost matrices
    Q1a = jnp.kron(jnp.eye(horizon), Q)
    Q3a = jnp.kron(jnp.eye(horizon), R)
    Q2a = QN

    Q1 = jnp.hstack(
        [Q1a, jnp.zeros((Q1a.shape[0], Q2a.shape[1])), jnp.zeros((Q1a.shape[0], Q3a.shape[1]))]
    )
    Q2 = jnp.hstack(
        [jnp.zeros((Q2a.shape[0], Q1a.shape[1])), Q2a, jnp.zeros((Q2a.shape[0], Q3a.shape[1]))]
    )
    Q3 = jnp.hstack(
        [jnp.zeros((Q3a.shape[0], Q1a.shape[1])), jnp.zeros((Q3a.shape[0], Q2a.shape[1])), Q3a]
    )

    Q_bar = jnp.vstack([Q1, Q2, Q3])
    # p_bar = jnp.zeros((Q1a.shape[0] + Q2a.shape[0] + horizon * n_inputs))

    Ax = jnp.kron(jnp.eye(horizon + 1), -jnp.eye(n_states)) + jnp.kron(
        jnp.eye(horizon + 1, k=-1), A
    )
    Bu = jnp.kron(jnp.vstack([jnp.zeros((1, horizon)), jnp.eye(horizon)]), B)

    A_eq = jnp.hstack([Ax, Bu])
    # b_eq = jnp.zeros(((1 + horizon) * n_states,))

    # - input and state constraints
    A_ineq = None
    b_ineq = None

    @jit
    def mpc_to_qp(
        concatenated_x_xr: Array,
    ) -> Tuple[Array, Array, Union[Array, None], Union[Array, None], Array, Array]:
        """Transforms a Discrete-Time, Linear, Time-Invariant, Model Predictive
        Control trajectory objective into a standard quadratic program.

        Args:
            concatenated_x_xr (Array): concatenation of current state vector (x) and reference path (xr)

        Returns
        -------
            Q_bar: quadratic cost matrix.
            p_bar: linear cost vector.
            A_ineq: linear constraint matrix.
            b_ineq: linear constraint vector.
            A_eq: quadratic constraint matrix.
            b_eq: quadratic constraint vector.

        """
        x0 = concatenated_x_xr[0, :n_states]

        p_bar = jnp.hstack(
            [
                -Q1a @ concatenated_x_xr[1:, :n_states].T.flatten(),
                -Q2a @ concatenated_x_xr[-1, :n_states].T.flatten(),
                jnp.zeros(horizon * n_inputs),
            ]
        )

        # Linear dynamics (equality constraints)
        b_eq = jnp.hstack([-x0, jnp.zeros((horizon) * n_states)])

        # Returning the transformed QP problem
        return (
            Q_bar,
            p_bar,
            A_ineq,
            b_ineq,
            A_eq,
            b_eq,
        )

    return mpc_to_qp


@jit
def mpc_to_qp(
    x0: Array, xr: Array, A: Array, B: Array, Q: Array, R: Array, QN: Array, horizon: int
) -> Tuple[Array, Array, Union[Array, None], Union[Array, None], Array, Array]:
    """Transforms a Discrete-Time, Linear, Time-Invariant, Model Predictive
    Control trajectory objective into a standard quadratic program.

    Arguments:
        x0: initial state
        A: linear drift matrix
        B: control input matrix
        Q: incremental state cost matrix
        R: incremental control cost matrix
        QN: terminal state cost matrix
        n_steps: length of finite horizon

    Returns
    -------
        H: quadratic cost matrix.
        f: linear cost vector.
        A: linear constraint matrix.
        b: linear constraint vector.
        G: quadratic constraint matrix.
        h: quadratic constraint vector.

    """
    xr = jnp.hstack([x0.reshape(-1, 1), xr])
    n_states, n_inputs = B.shape

    # Formulating the quadratic cost matrices
    Q1a = jnp.kron(jnp.eye(horizon), Q)
    Q3a = jnp.kron(jnp.eye(horizon), R)
    Q2a = QN

    Q1 = jnp.hstack(
        [Q1a, jnp.zeros((Q1a.shape[0], Q2a.shape[1])), jnp.zeros((Q1a.shape[0], Q3a.shape[1]))]
    )
    Q2 = jnp.hstack(
        [jnp.zeros((Q2a.shape[0], Q1a.shape[1])), Q2a, jnp.zeros((Q2a.shape[0], Q3a.shape[1]))]
    )
    Q3 = jnp.hstack(
        [jnp.zeros((Q3a.shape[0], Q1a.shape[1])), jnp.zeros((Q3a.shape[0], Q2a.shape[1])), Q3a]
    )

    Q_bar = jnp.vstack([Q1, Q2, Q3])
    p_bar = jnp.hstack(
        [
            -2 * Q1a @ xr[:, :-1].flatten(),
            -2 * Q2a @ xr[:, -1].flatten(),
            jnp.zeros(horizon * n_inputs),
        ]
    )

    Ax = jnp.kron(jnp.eye(horizon + 1), -jnp.eye(n_states)) + jnp.kron(
        jnp.eye(horizon + 1, k=-1), A
    )
    Bu = jnp.kron(jnp.vstack([jnp.zeros((1, horizon)), jnp.eye(horizon)]), B)

    # Linear dynamics (equality constraints)
    A_eq = jnp.hstack([Ax, Bu])
    b_eq = jnp.hstack([-x0, jnp.zeros((horizon) * n_states)])

    # - input and state constraints
    A_ineq = None
    b_ineq = None

    # Returning the transformed QP problem
    return (
        Q_bar,
        p_bar,
        A_ineq,
        b_ineq,
        A_eq,
        b_eq,
    )


def quadratic_mpc_solver(
    x0: Array, xr: Array, A: Array, B: Array, Q: Array, R: Array, QN: Array, N: int
) -> Tuple[Array, Array]:
    n = x0.shape[0]
    m = B.shape[1]
    # Convert Discrete-Time, LTI MPC problem into QP and solve
    H, f, A, b, G, h = mpc_to_qp(x0, xr, A, B, Q, R, QN, N)
    qp_result = _DEFAULT_SOLVER(H, f, A, b, G, h)
    sol = qp_result.primal
    if qp_result.status != 1:
        print("MPC Infeasible.")

    # Extract optimal state and control trajectories
    sol = jnp.array(sol).flatten()
    x_opt = sol[: (N + 1) * n].reshape((N + 1, n)).T
    u_opt = sol[(N + 1) * n :].reshape((N, m)).T

    return x_opt, u_opt
