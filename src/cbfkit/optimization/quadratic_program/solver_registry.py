"""Unified QP solver interface with runtime selection.

Provides a common ``QpSolution`` return type and factory functions that wrap
each backend (jaxopt, cvxopt, casadi) behind a single callable signature.

**Objective convention.** Every solver returned by :func:`get_solver` solves

.. math::

    \\min_x \\; x^T H x + f^T x \\quad \\text{s.t.} \\quad G x \\le h \\;(, A x = b)

This matches the CBF-CLF-QP generator, which passes ``f = -2 H u_nom`` so the
unconstrained minimizer is ``u_nom``.  Backends whose native form is
``min 1/2 x^T P x + q^T x`` (OSQP, CVXOPT, PDIPM) are adapted at this boundary
(``P = 2 H`` or equivalently ``q = f/2``); the raw modules
(``qp_solver_pdipm.solve_qp_pdipm``, ``qp_solver_cvxopt.solve``) keep their
native halved convention.

Usage::

    from cbfkit.optimization.quadratic_program import get_solver

    solver = get_solver("jaxopt", max_iter=5000, tol=1e-5)
    sol = solver(H, f, G, h)
    print(sol.primal, sol.status)

    # Or with warm-starting (jaxopt only):
    sol2 = solver(H, f, G, h, init_params=sol.params)
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from jax import Array

from cbfkit.utils.user_types.solvers import QpSolution, QpSolverCallable, with_solver_metadata

from ._validation import validate_qp_shapes

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

        validate_qp_shapes(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)

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

        status = jnp.where((status == 1) & ~jnp.all(jnp.isfinite(sol.primal)), 0, status)
        return QpSolution(primal=sol.primal, status=status, params=(sol, state))

    return with_solver_metadata(solve_with_details, name="jaxopt", jit_compatible=True)


def cvxopt_solver() -> QpSolverCallable:
    """Create a QP solver backed by CVXOPT.

    Not JIT-compatible.  Warm-starting is not supported (``init_params``
    is ignored and ``params`` is always ``None``).
    """
    from cbfkit.optimization.quadratic_program.qp_solver_cvxopt import (
        solve_with_details,
    )

    return with_solver_metadata(solve_with_details, name="cvxopt", jit_compatible=False)


def casadi_solver() -> QpSolverCallable:
    """Create a QP solver backed by CasADi / qpOASES.

    Not JIT-compatible.  Warm-starting is not supported (``init_params``
    is ignored and ``params`` is always ``None``).
    """
    from cbfkit.optimization.quadratic_program.qp_solver_casadi import (
        solve_with_details,
    )

    return with_solver_metadata(solve_with_details, name="casadi", jit_compatible=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def fast_solver(max_iter: Optional[int] = None, tol: float = 1e-6) -> QpSolverCallable:
    """Fast PDIPM solver for small CBF-CLF problems.

    Mehrotra predictor-corrector primal-dual interior-point method. Designed
    for the QP shapes that arise in CBF-CLF-QP safety filtering (2-8 variables,
    5-30 constraints). Robust on slack-relaxed problems where dual coordinate
    descent fails to converge.

    Under ``jax.jit`` its per-solve time is comparable to ``get_solver("jaxopt")``
    (OSQP) on typical CBF-QP sizes (slightly slower at 2x5, ~1.6x faster at 8x20);
    called eagerly it is ~600-800x faster only because OSQP's iterations are
    dispatched one by one (see ``benchmarks/qp_solver_comparison.py --jit``).
    JIT-compatible and warm-startable across consecutive control steps.

    Inequality constraints only: passing ``a_mat``/``b_vec`` raises
    ``NotImplementedError`` rather than dropping them.

    Args:
        max_iter: Maximum PDIPM iterations. ``None`` (default) defers to
            ``qp_solver_pdipm.DEFAULT_MAX_ITER`` so the solver's calibrated
            budget is not silently pinned here.
        tol: Combined primal/dual/complementarity residual tolerance.
    """
    from cbfkit.optimization.quadratic_program.qp_solver_pdipm import (
        DEFAULT_MAX_ITER,
        PdipmState,
        solve_qp_pdipm,
    )

    if max_iter is None:
        max_iter = DEFAULT_MAX_ITER

    def solve_with_details(
        h_mat: Array,
        f_vec: Array,
        g_mat: Optional[Array] = None,
        h_vec: Optional[Array] = None,
        a_mat: Optional[Array] = None,
        b_vec: Optional[Array] = None,
        init_params: Any = None,
    ) -> QpSolution:
        if a_mat is not None or b_vec is not None:
            raise NotImplementedError(
                "The 'fast' (PDIPM) backend solves inequality-constrained QPs only "
                "(min x'Hx + f'x s.t. Gx <= h), but was given equality constraints "
                "via a_mat/b_vec. Silently dropping them would return a solution "
                "that violates Ax = b — e.g. an MPC trajectory ignoring its own "
                "dynamics. Use get_solver('jaxopt') or get_solver('casadi'), which "
                "support equality constraints."
            )

        validate_qp_shapes(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)

        if g_mat is None or g_mat.shape[0] == 0:
            # Registry convention min x'Hx + f'x  =>  2H x* = -f.
            x = jnp.linalg.solve(2.0 * h_mat, -f_vec)
            return QpSolution(
                primal=x, status=jnp.where(jnp.all(jnp.isfinite(x)), 1, 0), params=None
            )

        # Extract warm-start state from previous QpSolution.params
        warm: Optional[PdipmState] = None
        if init_params is not None:
            if isinstance(init_params, tuple) and len(init_params) == 2:
                _, state = init_params
                if isinstance(state, PdipmState):
                    warm = state
            elif isinstance(init_params, PdipmState):
                warm = init_params

        # Convention adapter: solve_qp_pdipm natively solves
        # min 1/2 x'Px + q'x; the registry convention is min x'Hx + f'x,
        # so pass P = 2H (the jaxopt wrapper adapts via q = f/2 instead).
        sol, status, state = solve_qp_pdipm(
            2.0 * h_mat,
            f_vec,
            g_mat,
            h_vec,
            warm_start=warm,
            max_iter=max_iter,
            tol=tol,
        )
        if warm is not None:
            # Cold-restart fallback: a warm start from the previous step is occasionally a
            # *bad* starting point -- e.g. when the nominal input or the active set jumps
            # (an MPPI replan boundary) the carried (s, lam) can be nearly degenerate and
            # the budget runs out before recovery, while the cold start converges. Fire on
            # a non-finite "solution" too (a degenerate warm start can NaN the iterate in a
            # way the residual check misclassifies). Runtime cost is one extra solve only on
            # failing steps (lax.cond executes one branch).
            def _cold(_):
                return solve_qp_pdipm(
                    2.0 * h_mat, f_vec, g_mat, h_vec, warm_start=None, max_iter=max_iter, tol=tol
                )

            def _keep(_):
                return sol, status, state

            bad = (status != 1) | ~jnp.all(jnp.isfinite(sol))
            sol, status, state = jax.lax.cond(bad, _cold, _keep, None)
        status = jnp.where((status == 1) & ~jnp.all(jnp.isfinite(sol)), 0, status)
        return QpSolution(primal=sol, status=status, params=(sol, state))

    return with_solver_metadata(solve_with_details, name="fast", jit_compatible=True)


_SOLVER_FACTORIES: dict[str, Callable[..., QpSolverCallable]] = {
    "jaxopt": jaxopt_solver,
    "cvxopt": cvxopt_solver,
    "casadi": casadi_solver,
    "fast": fast_solver,
}


def get_solver(name: str = "jaxopt", **kwargs) -> QpSolverCallable:
    """Look up a QP solver by name and return a configured callable.

    Args:
        name: One of ``"jaxopt"``, ``"cvxopt"``, ``"casadi"``, ``"fast"``.
        **kwargs: Forwarded to the solver factory (e.g. ``max_iter``,
            ``tol`` for jaxopt/fast).

    Returns:
        A callable with signature
        ``(H, f, G, h, A, b, init_params) -> QpSolution``.

    Raises:
        KeyError: If *name* is not a registered solver.

    Environment override:
        When ``CBFKIT_QP_SOLVER`` is set, a request for the default solver
        (``name="jaxopt"``) is rerouted to the named solver. Used by the
        integration test suite to run every example/tutorial under both
        ``jaxopt`` and ``fast`` without modifying the scripts themselves.
        Explicit non-default requests (e.g. ``get_solver("cvxopt")``) are
        not affected.
    """
    import os

    overridden = False
    if name == "jaxopt":
        override = os.environ.get("CBFKIT_QP_SOLVER", "").strip().lower()
        if override and override != "jaxopt":
            name = override
            overridden = True

    if name not in _SOLVER_FACTORIES:
        available = ", ".join(sorted(_SOLVER_FACTORIES))
        raise KeyError(f"Unknown QP solver {name!r}. Available: {available}")
    factory = _SOLVER_FACTORIES[name]
    if overridden:
        signature = inspect.signature(factory)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        unsupported = set(kwargs) - signature.parameters.keys() if not accepts_kwargs else set()
        if unsupported:
            warnings.warn(
                f"Solver override {name!r} ignores unsupported options: "
                + ", ".join(sorted(unsupported)),
                UserWarning,
                stacklevel=2,
            )
            kwargs = {key: value for key, value in kwargs.items() if key not in unsupported}
    # A TypeError raised inside the factory is an actual failure, never a retry signal.
    return factory(**kwargs)


def list_solvers() -> list[str]:
    """Return the names of all registered QP solvers."""
    return sorted(_SOLVER_FACTORIES)
