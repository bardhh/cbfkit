"""Quadratic program solver using the Jaxopt library."""

from typing import Any, Optional, Tuple, Union

# import jax.numpy as jnp
from jax import Array, jit
from jaxopt import OSQP, EqualityConstrainedQP

# Instantiate QP solver objects
QP = OSQP(maxiter=1000000, tol=1e-3)
EC_QP = EqualityConstrainedQP()


def solve_equality_constrained_qp(
    params_obj: Tuple[Array, Array],
    params_eq: Tuple[Array, Array],
) -> Tuple[Array, int]:
    """Solves an equality constrained quadratic program.

    Args:
        params_obj (Tuple[Array, Array]): Quadratic cost parameters (P, q).
        params_eq (Tuple[Array, Array]): Equality constraint parameters (A, b).

    Returns:
        Tuple[Array, int]: The solution to the QP and a status code.
    """
    solution = EC_QP.run(
        params_obj=params_obj,
        params_eq=params_eq,
    )
    return solution.params.primal, solution.state is None


def solve_inequality_constrained_qp(
    params_obj: Tuple[Array, Array],
    params_eq: Optional[Tuple[Array, Array]],
    params_ineq: Optional[Tuple[Array, Array]],
    init_params: Optional[Any] = None,
) -> Tuple[Array, int, Any]:
    """Solves an inequality constrained quadratic program.

    Args:
        params_obj (Tuple[Array, Array]): Quadratic cost parameters (P, q).
        params_eq (Optional[Tuple[Array, Array]]): Optional equality constraint parameters (A, b).
        params_ineq (Optional[Tuple[Array, Array]]): Optional inequality constraint parameters (G, h).
        init_params (Optional[Any]): Optional initial parameters for warm-starting.

    Returns:
        Tuple[Array, int, Any]: The solution to the QP, a status code, and the solver parameters.
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
    # We return the raw status code to allow callers to inspect failure reason.
    return sol.primal, status, (sol, state)


@jit
def solve_with_details(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
    init_params: Optional[Any] = None,
) -> Tuple[Array, int, Any]:
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
        (sol, status, params): Solution, raw status (int), and solver parameters.
    """
    params_obj = (h_mat, 0.5 * f_vec)
    params_eq = None if (a_mat is None or b_vec is None) else (a_mat, b_vec)
    params_ineq = None if (g_mat is None or h_vec is None) else (g_mat, h_vec)

    sol, status, params = solve_inequality_constrained_qp(
        params_obj, params_eq, params_ineq, init_params
    )

    return sol, status, params


@jit
def solve_with_state(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
    init_params: Optional[Any] = None,
) -> Tuple[Array, int, Any]:
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
    success = status == 1
    return sol, success, params


@jit
def solve(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
) -> Tuple[Array, int]:
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
