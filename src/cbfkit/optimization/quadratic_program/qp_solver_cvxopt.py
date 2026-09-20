"""Quadratic program solver using the CVXOPT library."""

import platform
from importlib import import_module
from types import ModuleType
from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import Array

from cbfkit.utils.user_types.solvers import QpSolution

# Resolve the optional, platform-specific backend once at import time. Keep its
# native matrix types behind the module boundary: cvxopt and kvxopt ship
# different stubs even though both accept NumPy arrays at runtime.
_cvxopt_backend: ModuleType | None = None
_cvxopt_error: ImportError | None = None
try:
    _mach = platform.machine().lower()
    _cvxopt_backend = import_module("kvxopt" if "arm" in _mach or "aarch" in _mach else "cvxopt")
except ImportError as _e:
    _cvxopt_error = _e


def _ensure_cvxopt() -> ModuleType:
    if _cvxopt_backend is None:
        raise ImportError(
            "To use the cvxopt solver, please install the 'cvxopt' extra: "
            "pip install 'cbfkit[cvxopt]'"
        ) from _cvxopt_error
    return _cvxopt_backend


def solve(
    p_mat: Array,
    q_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
) -> Tuple[Array, int]:
    """Solve a quadratic program using the cvxopt solver.

    Args:
        p_mat: quadratic cost matrix (n_vars, n_vars)
        q_vec: linear cost vector (n_vars,)
        g_mat: inequality constraint matrix (n_ineq, n_vars)
        h_vec: inequality constraint bounds (n_ineq,)
        a_mat: equality constraint matrix (n_eq, n_vars)
        b_vec: equality constraint bounds (n_eq,)

    Returns
    -------
        (sol, success): Solution array and boolean success flag.
    """
    backend = _ensure_cvxopt()
    matrix, solvers = backend.matrix, backend.solvers

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
    check_matrix = np.vstack([item for item in [p_mat, g_mat, a_mat] if item is not None])
    if np.linalg.matrix_rank(check_matrix) < np.array(p_mat).shape[0]:
        raise ValueError("Ill-posed problem: Rank([H; G; A]) < number of decision variables")

    # Compute solution
    sol: Dict[str, Any] = solvers.qp(
        p_mat, q_vec, G=g_mat, h=h_vec, A=a_mat, b=b_vec, options=options
    )

    success: bool = sol["status"] == "optimal"
    if not success:
        if sol["status"] == "unknown":
            success = bool(np.all(np.array(g_mat) @ np.array(sol["x"]) - np.array(h_vec) <= 0))

    return jnp.array(sol["x"]).reshape((len(sol["x"]),)), success


def solve_with_details(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
    init_params: Any = None,
) -> QpSolution:
    """Solve a QP using CVXOPT, returning a unified :class:`QpSolution`.

    ``init_params`` is accepted for interface compatibility but ignored
    (CVXOPT does not support warm-starting).

    Note: the registry convention is ``min x'Hx + f'x`` while CVXOPT's native
    form is ``min 1/2 x'Px + q'x``; the ``P = 2H`` conversion happens here so
    all ``get_solver()`` backends agree (see ``solver_registry`` docstring).
    """
    primal, success = solve(2.0 * h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)
    return QpSolution(primal=primal, status=1 if success else 0, params=None)
