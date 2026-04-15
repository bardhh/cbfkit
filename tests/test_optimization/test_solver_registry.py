"""Tests for the unified QP solver registry."""

import pytest
import jax.numpy as jnp

from cbfkit.optimization.quadratic_program.solver_registry import (
    QpSolution,
    get_solver,
    list_solvers,
    jaxopt_solver,
)


# -- Shared test problem ------------------------------------------------


def _feasible_problem():
    """min ‖x‖²  s.t.  x >= -1  (solution: x = [0, 0])."""
    H = 2.0 * jnp.eye(2)
    f = jnp.zeros(2)
    G = -jnp.eye(2)
    h = jnp.ones(2)
    return H, f, G, h


def _feasible_problem_with_cost():
    """min 0.5 x'Hx + 0.5 f'x  s.t.  x <= 3  (solution: x = [1, 2]).

    The registry solver applies 0.5*f internally, so we pass f = -4*target
    to get the intended minimizer.
    """
    H = 2.0 * jnp.eye(2)
    f = -4.0 * jnp.array([1.0, 2.0])
    G = jnp.eye(2)
    h = 3.0 * jnp.ones(2)
    return H, f, G, h


# -- Registry tests ------------------------------------------------------


class TestRegistry:
    def test_list_solvers(self):
        names = list_solvers()
        assert "jaxopt" in names
        assert "cvxopt" in names
        assert "casadi" in names

    def test_get_unknown_solver_raises(self):
        with pytest.raises(KeyError, match="Unknown QP solver"):
            get_solver("nonexistent")

    def test_get_jaxopt_default(self):
        solver = get_solver()
        assert getattr(solver, "solver_name", None) == "jaxopt"
        assert getattr(solver, "jit_compatible", False) is True

    def test_get_jaxopt_custom_params(self):
        solver = get_solver("jaxopt", max_iter=500, tol=1e-3)
        assert getattr(solver, "solver_name", None) == "jaxopt"


# -- QpSolution tests ----------------------------------------------------


class TestQpSolution:
    def test_tuple_unpacking(self):
        sol = QpSolution(primal=jnp.array([1.0]), status=1, params=None)
        primal, status, params = sol
        assert float(primal[0]) == 1.0
        assert status == 1
        assert params is None

    def test_indexing(self):
        sol = QpSolution(primal=jnp.array([1.0]), status=1, params="state")
        assert sol[0] is sol.primal
        assert sol[1] == 1
        assert sol[2] == "state"

    def test_repr(self):
        sol = QpSolution(primal=jnp.array([1.0]), status=1)
        assert "QpSolution" in repr(sol)


# -- Jaxopt solver tests --------------------------------------------------


class TestJaxoptSolver:
    def test_feasible(self):
        solver = get_solver("jaxopt")
        H, f, G, h = _feasible_problem()
        sol = solver(H, f, G, h)
        assert sol.status == 1
        assert jnp.allclose(sol.primal, jnp.zeros(2), atol=1e-3)

    def test_feasible_with_cost(self):
        solver = get_solver("jaxopt")
        H, f, G, h = _feasible_problem_with_cost()
        sol = solver(H, f, G, h)
        assert sol.status == 1
        assert jnp.allclose(sol.primal, jnp.array([1.0, 2.0]), atol=1e-3)

    def test_warm_start(self):
        solver = get_solver("jaxopt")
        H, f, G, h = _feasible_problem_with_cost()
        sol1 = solver(H, f, G, h)
        assert sol1.params is not None

        sol2 = solver(H, f, G, h, init_params=sol1.params)
        assert sol2.status == 1
        assert jnp.allclose(sol1.primal, sol2.primal, atol=1e-3)

    def test_custom_max_iter(self):
        solver = jaxopt_solver(max_iter=100, tol=1e-4)
        H, f, G, h = _feasible_problem()
        sol = solver(H, f, G, h)
        # Should still solve this simple problem
        assert sol.status == 1

    def test_jit_compatible_flag(self):
        solver = get_solver("jaxopt")
        assert solver.jit_compatible is True

    def test_input_validation(self):
        solver = get_solver("jaxopt")
        with pytest.raises(ValueError, match="1D array"):
            solver(jnp.eye(2), jnp.zeros((2, 1)))  # f_vec must be 1D

    def test_equality_constraints(self):
        solver = get_solver("jaxopt")
        H = 2.0 * jnp.eye(2)
        f = jnp.zeros(2)
        # x1 + x2 = 1
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([1.0])
        sol = solver(H, f, a_mat=A, b_vec=b)
        assert sol.status == 1
        assert jnp.allclose(sol.primal[0] + sol.primal[1], 1.0, atol=1e-3)


# -- Backward compatibility -----------------------------------------------


class TestBackwardCompatibility:
    """Ensure existing direct imports still work."""

    def test_legacy_solve_import(self):
        from cbfkit.optimization.quadratic_program import solve

        H, f, G, h = _feasible_problem()
        x, success = solve(H, f, G, h)
        assert success
        assert jnp.allclose(x, jnp.zeros(2), atol=1e-3)

    def test_legacy_solve_with_details_import(self):
        from cbfkit.optimization.quadratic_program.qp_solver_jaxopt import (
            solve_with_details,
        )

        H, f, G, h = _feasible_problem()
        sol = solve_with_details(H, f, G, h)
        # Legacy QpSolution is a NamedTuple — supports unpacking
        primal, status, params = sol
        assert status == 1

    def test_qp_solution_from_init(self):
        from cbfkit.optimization.quadratic_program import QpSolution as PubQpSolution

        sol = PubQpSolution(primal=jnp.array([1.0]), status=1)
        assert sol.status == 1
