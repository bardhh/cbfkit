"""Quadratic program solver using the Jaxopt library."""

from typing import Any, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array, jit
from jaxopt import OSQP, EqualityConstrainedQP

from cbfkit.utils.user_types import SolverStatus

# Instantiate QP solver objects
MAX_ITER = 1000000
QP = OSQP(maxiter=MAX_ITER, tol=1e-3)
EC_QP = EqualityConstrainedQP()


class QpSolution(NamedTuple):
    """Return type for QP solvers."""

    primal: Array
    status: int
    params: Tuple[Any, Any]


class EqualityQpSolution(NamedTuple):
    """Return type for equality-constrained QP solvers."""

    primal: Array
    success: bool


def solve_equality_constrained_qp(
    params_obj: Tuple[Array, Array],
    params_eq: Tuple[Array, Array],
) -> EqualityQpSolution:
    """Solves an equality constrained quadratic program.

    Args:
        params_obj (Tuple[Array, Array]): Quadratic cost parameters (P, q).
        params_eq (Tuple[Array, Array]): Equality constraint parameters (A, b).

    Returns:
        EqualityQpSolution: The solution to the QP and a status code.
    """
    solution = EC_QP.run(
        params_obj=params_obj,
        params_eq=params_eq,
    )
    return EqualityQpSolution(
        primal=solution.params.primal,
        success=solution.state is None,
    )


def solve_inequality_constrained_qp(
    params_obj: Tuple[Array, Array],
    params_eq: Optional[Tuple[Array, Array]],
    params_ineq: Optional[Tuple[Array, Array]],
    init_params: Optional[Any] = None,
) -> QpSolution:
    """Solves an inequality constrained quadratic program.

    Args:
        params_obj (Tuple[Array, Array]): Quadratic cost parameters (P, q).
        params_eq (Optional[Tuple[Array, Array]]): Optional equality constraint parameters (A, b).
        params_ineq (Optional[Tuple[Array, Array]]): Optional inequality constraint parameters (G, h).
        init_params (Optional[Any]): Optional initial parameters for warm-starting.

    Returns:
        QpSolution: The solution to the QP, a status code, and the solver parameters.
    """
    # Handle unpacked init_params if it comes from a previous run of this function
    real_init_params = init_params
    if isinstance(init_params, tuple) and len(init_params) == 2:
        real_init_params = init_params[0]

    sol, state = QP.run(
        init_params=real_init_params,
        params_obj=params_obj,
        params_eq=params_eq,
        params_ineq=params_ineq,
    )
    status = state.status

    # Sentinel: Detect ambiguous MAX_ITER_REACHED (status=0 but iter_num >= maxiter)
    # This happens when jaxopt gives up but hasn't flagged it as 2 explicitly.
    # We map it to 5 (MAX_ITER_REACHED (UNSOLVED)) to distinguish it from a successful status (2).
    # This allows downstream controllers to fail safely (since 5 != success) but report the specific cause.
    status = jnp.where(
        (status == 0) & (state.iter_num >= MAX_ITER),
        SolverStatus.MAX_ITER_UNSOLVED,
        status,
    )

    # We return the raw status code to allow callers to inspect failure reason.
    return QpSolution(primal=sol.primal, status=status, params=(sol, state))


@jit
def solve_with_details(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
    init_params: Optional[Any] = None,
) -> QpSolution:
    """Solve a quadratic program using the jaxopt solver, returning full details.

    Args:
        h_mat: quadratic cost matrix
        f_vec: linear cost vector
        g_mat: linear inequality constraint matrix
        b_vec: linear inequality constraint vector
        g_mat: linear equality constraint matrix
        h_vec: linear equality constraint vector
        init_params: Optional initial parameters for warm-starting.

    Returns
    -------
        QpSolution: Solution, raw status (int), and solver parameters.
    """
    params_obj = (h_mat, 0.5 * f_vec)
    params_eq = None if (a_mat is None or b_vec is None) else (a_mat, b_vec)
    params_ineq = None if (g_mat is None or h_vec is None) else (g_mat, h_vec)

    return solve_inequality_constrained_qp(params_obj, params_eq, params_ineq, init_params)


@jit
def solve_with_state(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
    init_params: Optional[Any] = None,
) -> Tuple[Array, bool, Any]:
    """Solve a quadratic program using the jaxopt solver, returning solver state.

    Note: Returns status as a boolean success flag (True if status == 1).

    Args:
        h_mat: quadratic cost matrix
        f_vec: linear cost vector
        g_mat: linear inequality constraint matrix
        b_vec: linear inequality constraint vector
        g_mat: linear equality constraint matrix
        h_vec: linear equality constraint vector
        init_params: Optional initial parameters for warm-starting.

    Returns
    -------
        (sol, success, params): Solution, success flag (bool), and solver parameters.
    """
    sol, status, params = solve_with_details(
        h_mat, f_vec, g_mat, h_vec, a_mat, b_vec, init_params
    )
    # Only SOLVED (1) is success. MAX_ITER_REACHED (2) may return non-converged solution.
    success = status == SolverStatus.SOLVED
    return sol, success, params


@jit
def solve(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
) -> Tuple[Array, bool]:
    """Solve a quadratic program using the jaxopt solver.

    Args:
        h_mat: quadratic cost matrix
        f_vec: linear cost vector
        g_mat: linear inequality constraint matrix
        b_vec: linear inequality constraint vector
        g_mat: linear equality constraint matrix
        h_vec: linear equality constraint vector

    Returns
    -------
        sol['x']: Solution to the QP
    """
    sol, success, _ = solve_with_state(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)
    return sol, success
