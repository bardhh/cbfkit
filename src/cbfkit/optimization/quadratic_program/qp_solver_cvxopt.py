"""
qp_solver_cvxopt
================

This module implements a solver for quadratic programs using the CVXOPT library.

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
>>> import qp_solver_cvxopt
>>> sol, status = qp_solver_cvxopt.solve(
        h_mat=jnp.eye(2),
        f_vec=jnp.ones((2,)),
        g_mat=jnp.ones((2, 1)),
        h_vec=jnp.array([1.0]),
        a_mat=None,
        b_vec=None,
    )

"""
import platform
from typing import Union, Tuple, Dict, Any
import jax.numpy as jnp
import numpy as np
from jax import Array


mach = platform.machine().lower()
if "arm" in mach or "aarch" in mach:
    # pylint: disable=E0401
    from kvxopt import matrix, solvers  # type: ignore[reportMissingImports]
else:
    from cvxopt import matrix, solvers


def solve(
    p_mat: Array,
    q_vec: Array,
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
    # Use the cvxopt library to solve the quadratic program
    p_mat = matrix(np.array(p_mat, dtype=float))
    q_vec = matrix(np.array(q_vec, dtype=float))
    options = {"show_progress": False}

    # Inequality constraints
    if g_mat is not None and h_vec is not None:
        g_mat = matrix(np.array(g_mat, dtype=float))
        h_vec = matrix(np.array(h_vec, dtype=float))

    # Equality constraints
    if a_mat is not None and b_vec is not None:
        if np.linalg.matrix_rank(np.array(a_mat)) < np.array(a_mat).shape[0]:
            raise ValueError("Ill-posed problem: Rank(A) < number of equality constraints")
        a_mat = matrix(np.array(a_mat, dtype=float))
        b_vec = matrix(np.array(b_vec, dtype=float))

    # Check problem conditioning
    check_matrix = np.vstack(list(filter(lambda item: item is not None, [p_mat, g_mat, a_mat])))
    if np.linalg.matrix_rank(check_matrix) < np.array(p_mat).shape[0]:
        raise ValueError("Ill-posed problem: Rank([H; G; A]) < number of decision variables")

    # Compute solution
    sol: Dict[str, Any] = solvers.qp(
        p_mat, q_vec, G=g_mat, h=h_vec, A=a_mat, b=b_vec, options=options
    )

    success: bool = sol["status"] == "optimal"
    if not success:
        if sol["status"] == "unknown":
            success: bool = np.all(np.array(g_mat) @ np.array(sol["x"]) - np.array(h_vec) <= 0)

    return jnp.array(sol["x"]).reshape((len(sol["x"]),)), success
