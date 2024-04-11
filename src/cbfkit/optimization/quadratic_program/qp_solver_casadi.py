"""
qp_solver_casadi
================

This module implements a solver for quadratic programs using the Casadi library.

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

from typing import Union, Tuple
import casadi as ca
import numpy as np
from jax import Array
import jax.numpy as jnp
import contextlib
import os


def solve(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
) -> Tuple[Array, int]:
    """
    Solve a quadratic program using the Casadi solver.

    Args:
        h_mat: quadratic cost matrix
        f_vec: linear cost vector
        g_mat: linear inequality constraint matrix
        b_vec: linear inequality constraint vector
        g_mat: linear equality constraint matrix
        h_vec: linear equality constraint vector

    Returns:
        solution: Solution to the QP
        status: True if optimal solution found

    """
    # Define decision variables
    n = len(f_vec)
    x = ca.MX.sym("x", n)

    # Format vectors and matrices in Casadi
    h_mat = ca.MX(np.array(h_mat))
    f_vec = ca.MX(np.array(f_vec))

    # Construct inequality constraints
    lbg_i = []
    ubg_i = []
    inequality_constraints = []
    if g_mat is not None and h_vec is not None:
        lbg_i += [-ca.inf for ii in range(len(h_vec))]
        ubg_i += [0 for ii in range(len(h_vec))]
        g_mat = ca.MX(np.array(g_mat))
        h_vec = ca.MX(np.array(h_vec))
        inequality_constraints += [ca.mtimes(g_mat, x) - h_vec]

    # Construct equality constraints
    lbg_e = []
    ubg_e = []
    equality_constraints = []
    if a_mat is not None and b_vec is not None:
        lbg_e += [0 for ii in range(len(b_vec))]
        ubg_e += [0 for ii in range(len(b_vec))]
        a_mat = ca.MX(np.array(a_mat))
        b_vec = ca.MX(np.array(b_vec))
        equality_constraints += [ca.mtimes(a_mat, x) - b_vec]

    # Define quadratic objective function
    objective = ca.mtimes(x.T, ca.mtimes(h_mat, x)) + ca.mtimes(f_vec.T, x)

    # Combine constraints
    lbg = lbg_i + lbg_e
    ubg = ubg_i + ubg_e
    constraints = inequality_constraints + equality_constraints

    # Define problem
    prob = {"f": objective, "x": x, "g": ca.vertcat(*constraints)}

    # # Set options for the solver
    solver_opts = {"printLevel": "none", "error_on_fail": False}

    # Solve the optimization problem
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        solver = ca.qpsol("solver", "qpoases", prob, solver_opts)
        solution = solver(lbg=lbg, ubg=ubg)
        success = solver.stats()["success"]

    return success * jnp.array(solution["x"]).reshape((n,)), success


if __name__ == "__main__":
    sol, status = solve(
        h_mat=jnp.eye(2),
        f_vec=jnp.array([0.0, -100.0]),
        g_mat=jnp.ones((2, 1)),
        h_vec=jnp.array([1.0]),
        a_mat=jnp.ones((2, 1)),
        b_vec=jnp.array([5.0]),
    )
    print(f"Solution: {sol}")
    print(f"Status: {status}")
