"""Tests for the fast small-QP solver and its integration with CBF-CLF-QP."""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from cbfkit.optimization.quadratic_program.qp_solver_fast import solve_qp_fast
from cbfkit.optimization.quadratic_program import get_solver, list_solvers


class TestFastQpSolver:
    """Core solver correctness tests."""

    def test_unconstrained_minimum(self):
        P = jnp.eye(2)
        q = jnp.array([-2.0, -4.0])
        G = jnp.zeros((1, 2))
        h = jnp.array([100.0])
        sol, status, _ = solve_qp_fast(P, q, G, h)
        assert jnp.allclose(sol, jnp.array([2.0, 4.0]), atol=1e-4)

    def test_box_constrained(self):
        P = jnp.eye(2)
        q = jnp.array([-10.0, -10.0])
        G = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1.0]])
        h = jnp.array([1.0, 1.0, 1.0, 1.0])
        sol, status, _ = solve_qp_fast(P, q, G, h)
        assert jnp.allclose(sol, jnp.array([1.0, 1.0]), atol=1e-4)

    def test_cbf_like_constraint(self):
        """Typical CBF-QP: minimize deviation from nominal subject to safety."""
        P = jnp.diag(jnp.array([1.0, 1.0, 2000.0]))
        q = jnp.array([-1.0, 0.0, 0.0])  # nominal = [1, 0]
        G = jnp.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [-0.8, -0.2, -1.0],  # CBF constraint
            ]
        )
        h = jnp.array([1.0, 1.0, 1.0, 1.0, 0.1])
        sol, status, _ = solve_qp_fast(P, q, G, h)
        assert int(status) == 1
        assert jnp.all(G @ sol <= h + 1e-4)

    def test_warm_start_reuse(self):
        P = jnp.eye(2)
        q = jnp.array([-0.5, -0.5])
        G = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1.0]])
        h = jnp.array([1.0, 1.0, 1.0, 1.0])
        sol1, _, ws = solve_qp_fast(P, q, G, h)
        sol2, _, _ = solve_qp_fast(P, q, G, h, warm_start=ws)
        assert jnp.allclose(sol1, sol2, atol=1e-6)

    def test_feasibility(self):
        """All solutions must satisfy constraints."""
        key = random.PRNGKey(0)
        for _ in range(10):
            key, k1, k2, k3, k4 = random.split(key, 5)
            n = 4
            m = 8
            P = jnp.diag(jnp.abs(random.normal(k1, (n,))) + 0.1)
            q = random.normal(k2, (n,))
            G = random.normal(k3, (m, n))
            h = jnp.abs(random.normal(k4, (m,))) + 0.5
            sol, _, _ = solve_qp_fast(P, q, G, h)
            assert jnp.all(
                G @ sol <= h + 1e-3
            ), f"Infeasible: max viol={float(jnp.max(G @ sol - h))}"

    def test_matches_jaxopt(self):
        """Objective value should match OSQP within tolerance."""
        from jaxopt import OSQP

        key = random.PRNGKey(42)
        for _ in range(10):
            key, k1, k2, k3, k4 = random.split(key, 5)
            n = int(random.randint(k1, (), 2, 6))
            m = int(random.randint(k2, (), n + 1, 3 * n))
            P = jnp.diag(jnp.abs(random.normal(k3, (n,))) + 0.1)
            key, k5, k6 = random.split(key, 3)
            q = random.normal(k5, (n,))
            G = random.normal(k6, (m, n))
            h_vec = jnp.abs(random.normal(k4, (m,))) + 0.5

            sol_fast, _, _ = solve_qp_fast(P, q, G, h_vec, max_iter=100)
            osqp = OSQP(tol=1e-6, maxiter=20000)
            osqp_sol, _ = osqp.run(params_obj=(P, q), params_ineq=(G, h_vec))

            obj_fast = float(0.5 * sol_fast @ P @ sol_fast + q @ sol_fast)
            obj_osqp = float(0.5 * osqp_sol.primal @ P @ osqp_sol.primal + q @ osqp_sol.primal)
            rel_err = abs(obj_fast - obj_osqp) / max(abs(obj_osqp), 1e-6)
            assert (
                rel_err < 0.01
            ), f"Objective mismatch: fast={obj_fast}, osqp={obj_osqp}, err={rel_err}"


class TestFastSolverRegistry:
    """Registry integration tests."""

    def test_fast_in_list(self):
        assert "fast" in list_solvers()

    def test_get_fast_solver(self):
        solver = get_solver("fast")
        assert solver.jit_compatible
        assert solver.solver_name == "fast"

    def test_fast_solver_via_registry(self):
        solver = get_solver("fast")
        P = jnp.eye(2)
        q = jnp.array([-2.0, -4.0])
        G = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1.0]])
        h = jnp.array(
            [3.0, 3.0, 5.0, 5.0]
        )  # box [-3,3] x [-5,5] — unconstrained opt [2,4] is feasible
        sol = solver(P, q, G, h)
        assert jnp.allclose(sol.primal, jnp.array([2.0, 4.0]), atol=1e-4)
        assert int(sol.status) == 1

    def test_fast_solver_warm_start_via_registry(self):
        solver = get_solver("fast")
        P = jnp.eye(2)
        q = jnp.array([-0.5, -0.3])
        G = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1.0]])
        h = jnp.array([1.0, 1.0, 1.0, 1.0])
        sol1 = solver(P, q, G, h)
        sol2 = solver(P, q, G, h, init_params=sol1.params)
        assert jnp.allclose(sol1.primal, sol2.primal, atol=1e-6)


class TestFastSolverCbfIntegration:
    """End-to-end test with CBF-CLF-QP controller."""

    def test_cbf_qp_with_fast_solver(self):
        from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
        from cbfkit.certificates import generate_certificate
        from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import (
            linear_class_k,
        )
        from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator
        from cbfkit.utils.user_types import ControllerData

        dynamics = two_dimensional_single_integrator()

        def h(x):
            return (x[0] - 2.0) ** 2 + (x[1]) ** 2 - 0.25

        barriers = generate_certificate(h, linear_class_k(1.0), input_style="state")

        controller = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([1.0, 1.0]),
            dynamics_func=dynamics,
            barriers=barriers,
            solver=get_solver("fast"),
        )

        key = random.PRNGKey(0)
        state = jnp.array([0.0, 0.0])
        u_nom = jnp.array([1.0, 0.0])
        data = ControllerData()

        u, data = controller(0.0, state, u_nom, key, data)
        assert u.shape == (2,)
        assert not data.error

    def test_fast_solver_matches_jaxopt_in_cbf(self):
        """Same CBF-QP produces same control with fast vs jaxopt solver."""
        from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
        from cbfkit.certificates import generate_certificate
        from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import (
            linear_class_k,
        )
        from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator
        from cbfkit.utils.user_types import ControllerData

        dynamics = two_dimensional_single_integrator()

        def h(x):
            return (x[0] - 1.5) ** 2 + (x[1]) ** 2 - 0.25

        barriers = generate_certificate(h, linear_class_k(1.0), input_style="state")

        ctrl_fast = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([1.0, 1.0]),
            dynamics_func=dynamics,
            barriers=barriers,
            solver=get_solver("fast"),
        )
        ctrl_osqp = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([1.0, 1.0]),
            dynamics_func=dynamics,
            barriers=barriers,
            solver=get_solver("jaxopt"),
        )

        key = random.PRNGKey(0)
        state = jnp.array([1.0, 0.0])  # near obstacle
        u_nom = jnp.array([1.0, 0.0])  # heading toward it
        data = ControllerData()

        u_fast, _ = ctrl_fast(0.0, state, u_nom, key, data)
        u_osqp, _ = ctrl_osqp(0.0, state, u_nom, key, data)

        assert jnp.allclose(
            u_fast, u_osqp, atol=1e-2
        ), f"Control mismatch: fast={u_fast}, osqp={u_osqp}"
