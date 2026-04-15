"""Unified QP solver interface with runtime selection.

Provides a common ``QpSolution`` return type and factory functions that wrap
each backend (jaxopt, cvxopt, casadi) behind a single callable signature.

Usage::

    from cbfkit.optimization.quadratic_program import get_solver

    solver = get_solver("jaxopt", max_iter=5000, tol=1e-5)
    sol = solver(H, f, G, h)
    print(sol.primal, sol.status)

    # Or with warm-starting (jaxopt only):
    sol2 = solver(H, f, G, h, init_params=sol.params)
"""

from __future__ import annotations

from typing import Any, Optional, Union

import jax.numpy as jnp
from jax import Array

from cbfkit.utils.user_types.callables import QpSolverCallable


class QpSolution:
    """Return type for all QP solvers.

    Attributes:
        primal: Solution vector.
        status: Integer status code (1 = solved).
        params: Solver-specific state for warm-starting.  ``None`` for
            backends that do not support warm-starting.
    """

    __slots__ = ("primal", "status", "params")

    def __init__(self, primal: Array, status: int, params: Any = None):
        self.primal = primal
        self.status = status
        self.params = params

    # Support tuple unpacking: primal, status, params = solution
    def __iter__(self):
        return iter((self.primal, self.status, self.params))

    def __getitem__(self, idx):
        return (self.primal, self.status, self.params)[idx]

    def __repr__(self):
        return f"QpSolution(primal={self.primal}, status={self.status})"


# ---------------------------------------------------------------------------
# Factory functions — each returns a QpSolverCallable
# ---------------------------------------------------------------------------


def jaxopt_solver(
    max_iter: int = 10000,
    tol: float = 1e-4,
) -> QpSolverCallable:
    """Create a JIT-compatible QP solver backed by jaxopt OSQP.

    Args:
        max_iter: Maximum OSQP iterations.
        tol: Convergence tolerance.

    Returns:
        A ``QpSolverCallable`` that returns :class:`QpSolution` with
        warm-start ``params``.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="jaxopt")
        from jaxopt import OSQP

    from cbfkit.utils.jit_monitor import JitMonitor

    qp = OSQP(maxiter=max_iter, tol=tol)

    def solve_with_details(
        h_mat: Array,
        f_vec: Array,
        g_mat: Union[Array, None] = None,
        h_vec: Union[Array, None] = None,
        a_mat: Union[Array, None] = None,
        b_vec: Union[Array, None] = None,
        init_params: Optional[Any] = None,
    ) -> QpSolution:
        JitMonitor.increment("qp_solver_jaxopt.solve_with_details")

        if f_vec.ndim != 1:
            raise ValueError(
                f"Linear cost 'f_vec' must be a 1D array of shape (n_vars,), "
                f"but got {f_vec.shape}."
            )
        if h_mat.ndim != 2:
            raise ValueError(
                f"Quadratic cost 'h_mat' must be a 2D array of shape (n_vars, n_vars), "
                f"but got {h_mat.shape}."
            )

        params_obj = (h_mat, 0.5 * f_vec)
        params_eq = None if (a_mat is None or b_vec is None) else (a_mat, b_vec)
        params_ineq = None if (g_mat is None or h_vec is None) else (g_mat, h_vec)

        real_init_params = init_params
        if isinstance(init_params, tuple) and len(init_params) == 2:
            real_init_params = init_params[0]

        sol, state = qp.run(
            init_params=real_init_params,
            params_obj=params_obj,
            params_eq=params_eq,
            params_ineq=params_ineq,
        )

        status = state.status
        status = jnp.where(
            (status == 0) & (state.iter_num >= max_iter),
            5,  # MAX_ITER_UNSOLVED
            status,
        )

        return QpSolution(primal=sol.primal, status=status, params=(sol, state))

    solve_with_details.jit_compatible = True
    solve_with_details.solver_name = "jaxopt"
    return solve_with_details


def cvxopt_solver() -> QpSolverCallable:
    """Create a QP solver backed by CVXOPT.

    Not JIT-compatible.  Warm-starting is not supported (``init_params``
    is ignored and ``params`` is always ``None``).
    """
    from cbfkit.optimization.quadratic_program.qp_solver_cvxopt import (
        solve_with_details,
    )

    solve_with_details.jit_compatible = False
    solve_with_details.solver_name = "cvxopt"
    return solve_with_details


def casadi_solver() -> QpSolverCallable:
    """Create a QP solver backed by CasADi / qpOASES.

    Not JIT-compatible.  Warm-starting is not supported (``init_params``
    is ignored and ``params`` is always ``None``).
    """
    from cbfkit.optimization.quadratic_program.qp_solver_casadi import (
        solve_with_details,
    )

    solve_with_details.jit_compatible = False
    solve_with_details.solver_name = "casadi"
    return solve_with_details


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SOLVER_FACTORIES = {
    "jaxopt": jaxopt_solver,
    "cvxopt": cvxopt_solver,
    "casadi": casadi_solver,
}


def get_solver(name: str = "jaxopt", **kwargs) -> QpSolverCallable:
    """Look up a QP solver by name and return a configured callable.

    Args:
        name: One of ``"jaxopt"``, ``"cvxopt"``, ``"casadi"``.
        **kwargs: Forwarded to the solver factory (e.g. ``max_iter``,
            ``tol`` for jaxopt).

    Returns:
        A callable with signature
        ``(H, f, G, h, A, b, init_params) -> QpSolution``.

    Raises:
        KeyError: If *name* is not a registered solver.
    """
    if name not in _SOLVER_FACTORIES:
        available = ", ".join(sorted(_SOLVER_FACTORIES))
        raise KeyError(f"Unknown QP solver {name!r}. Available: {available}")
    return _SOLVER_FACTORIES[name](**kwargs)


def list_solvers() -> list[str]:
    """Return the names of all registered QP solvers."""
    return sorted(_SOLVER_FACTORIES)
