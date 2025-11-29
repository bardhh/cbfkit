"""Quadratic program solver using the Jaxopt library."""

from typing import Optional, Tuple, Union

# import jax.numpy as jnp
from jax import Array, jit
from jaxopt import OSQP, EqualityConstrainedQP

# Instantiate QP solver objects
QP = OSQP()
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
) -> Tuple[Array, int]:
    """Solves an inequality constrained quadratic program.

    Args:
        params_obj (Tuple[Array, Array]): Quadratic cost parameters (P, q).
        params_eq (Optional[Tuple[Array, Array]]): Optional equality constraint parameters (A, b).
        params_ineq (Optional[Tuple[Array, Array]]): Optional inequality constraint parameters (G, h).

    Returns:
        Tuple[Array, int]: The solution to the QP and a status code.
    """
    sol, state = QP.run(
        params_obj=params_obj,
        params_eq=params_eq,
        params_ineq=params_ineq,
    )
    status = state.status
    # 1: SOLVED, 2: MAX_ITER_REACHED
    # We consider both as success for now to avoid dropping control inputs
    return sol.primal, (status == 1) | (status == 2)


@jit
def solve(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
) -> Tuple[Array, int]:
    """Solve a quadratic program using the cvxopt solver.

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
    params_obj = (h_mat, 0.5 * f_vec)
    params_eq = None if (a_mat is None or b_vec is None) else (a_mat, b_vec)
    params_ineq = None if (g_mat is None or h_vec is None) else (g_mat, h_vec)

    sol, status = solve_inequality_constrained_qp(params_obj, params_eq, params_ineq)

    return status * sol, status
