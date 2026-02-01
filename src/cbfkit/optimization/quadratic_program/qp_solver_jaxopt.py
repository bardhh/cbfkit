"""
qp_solver_jax
================

This module implements a solver for quadratic programs using the Jax library.

Functions
---------
-solve(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec): calculates solution to quadratic program specified by args

Notes
-----
Quadratic Program takes the following form:
min 1/2 x.T @ h_mat @ x + f_vec @ x
subject to
g_mat @ x <= h_vec
a_mat @ x = b_vec

Examples
--------
>>> import qp_solver_jax
>>> sol, status = qp_solver_jax.solve(
        h_mat=jnp.eye(2),
        f_vec=jnp.ones((2,))
        g_mat=jnp.ones((2, 1))
        h_vec=jnp.array([1.0])
        a_mat=None,
        b_vec=None
    )

"""

# import jax.numpy as jnp
from jax import Array, jit, lax
from typing import Union, Tuple
from jaxopt import OSQP, EqualityConstrainedQP, BoxOSQP

# Instantiate QP solver objects
QP = OSQP()
EC_QP = EqualityConstrainedQP()


def solve_equality_constrained_qp(
    params_obj: Tuple[Array, Array],
    params_eq: Tuple[Array, Array],
) -> Tuple[Array, int]:
    solution = EC_QP.run(
        params_obj=params_obj,
        params_eq=params_eq,
    )
    return solution.params.primal, solution.state is None


def solve_inequality_constrained_qp(
    params_obj: Tuple[Array, Array],
    params_eq: Tuple[Array, Array],
    params_ineq: Tuple[Array, Array],
) -> Tuple[Array, int]:
    sol, state = QP.run(
        params_obj=params_obj,
        params_eq=params_eq,
        params_ineq=params_ineq,
    )
    status = state.status
    return sol.primal, status == True


@jit
def solve(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
) -> Tuple[Array, int]:
    """
    Solve a quadratic program using the cvxopt solver.

    Args:
        h_mat: quadratic cost matrix
        f_vec: linear cost vector
        g_mat: linear inequality constraint matrix
        b_vec: linear inequality constraint vector
        g_mat: linear equality constraint matrix
        h_vec: linear equality constraint vector

    Returns:
        sol['x']: Solution to the QP
    """
    params_obj = (h_mat, 0.5 * f_vec)
    params_eq = None if (a_mat is None or b_vec is None) else (a_mat, b_vec)
    params_ineq = None if (g_mat is None or h_vec is None) else (g_mat, h_vec)

    sol, status = solve_inequality_constrained_qp(params_obj, params_eq, params_ineq)

    return status * sol, status
