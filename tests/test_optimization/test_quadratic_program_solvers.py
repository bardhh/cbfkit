"""Legacy JAXopt tuple API regression tests.

Registry backend parity, equality feasibility, and infeasibility are covered in
``test_solver_convention_parity.py`` and ``test_solver_constraint_cases.py``.
These tests retain coverage for the native JAXopt solve/solve_with_state APIs.
"""

import os
import unittest

import pytest
import jax.numpy as jnp
from jax import random

import cbfkit.optimization.quadratic_program.qp_solver_jaxopt as qp_jaxopt
from cbfkit.utils.user_types import QpSolverCallable

KEY = random.PRNGKey(0)


@pytest.mark.slow
class TestQuadraticProgramSolvers(unittest.TestCase):
    """Takes care of unit tests intended to verify the intended performance of quadratic program
    solvers."""

    eps = 1e-2

    def test_qp_jaxopt_feasible_inequality_only(self):
        """Tests that the JAXOPT-based quadratic program solver computes the correct solution for a
        sequence of problems with inequality constraints only."""
        self._test_qp_feasible_inequality_only(qp_jaxopt.solve)

    def test_qp_jaxopt_feasible_equality_only(self):
        """Tests that the JAXOPT-based quadratic program solver computes the correct solution for a
        sequence of problems with equality constraints only."""
        self._test_qp_feasible_equality_only(qp_jaxopt.solve)

    def test_qp_jaxopt_feasible(self):
        """Tests that the JAXOPT-based quadratic program solver computes the correct solution for a
        sequence of problems with both inequality and equality constraints."""
        self._test_qp_feasible(qp_jaxopt.solve)

    def test_qp_jaxopt_infeasible(self):
        """Tests that the JAXOPT-based quadratic program solver correctly identifies that the posed
        inequality/equality constrained problem is infeasible."""
        self._test_qp_infeasible(qp_jaxopt.solve)

    def test_qp_jaxopt_feasible_with_state(self):
        """Tests that the JAXOPT-based quadratic program solver computes the correct solution
        and returns state when using solve_with_state."""
        self._test_qp_feasible_with_state(qp_jaxopt.solve_with_state)

    def _test_qp_feasible_inequality_only(self, solver: QpSolverCallable):
        """Tests that the quadratic program solver computes the correct solution for a sequence of
        problems with inequality constraints only."""
        n_tests = 30 if not os.getenv("CBFKIT_TEST_MODE") else 2

        import time

        start = time.time()

        for tt in range(n_tests):
            n_vars = tt + 2

            # Objective function
            h_mat = jnp.eye(n_vars)
            f_vec = -2 * jnp.matmul(h_mat, jnp.array([ii for ii in range(n_vars)]))

            # Inequality Constraints
            g_mat = jnp.vstack([jnp.eye(n_vars), -jnp.eye(n_vars)])
            h_vec = jnp.ones((2 * n_vars,))

            # Equality constraints
            a_mat = None
            b_vec = None

            x, status = solver(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)

            # Correct status
            self.assertTrue(status)

            # Solution satisfies inequality constraints
            self.assertTrue(
                jnp.sum((jnp.matmul(g_mat, x).flatten() - h_vec) > self.eps) == 0,
                f"Failed Inequality constraints: {g_mat} * {x} - {h_vec} < 0",
            )

            # Solution is optimal
            if jnp.sum(jnp.matmul(g_mat, (x + self.eps)) > h_vec) == 0:
                self.assertTrue(
                    x.T @ h_mat @ x + f_vec.T @ x
                    < (x + self.eps).T @ h_mat @ (x + self.eps) + f_vec.T @ (x + self.eps)
                )
            if jnp.sum(jnp.matmul(g_mat, (x - self.eps)) > h_vec) == 0:
                self.assertTrue(
                    x.T @ h_mat @ x + f_vec.T @ x
                    < (x - self.eps).T @ h_mat @ (x - self.eps) + f_vec.T @ (x - self.eps)
                )

        print(f"test_qp_feasible_inequality_only: {time.time() - start}")

    def _test_qp_feasible_equality_only(self, solver: QpSolverCallable):
        """Tests that the quadratic program solver computes the correct solution for a sequence of
        problems with equality constraints only."""
        n_tests = 30 if not os.getenv("CBFKIT_TEST_MODE") else 2

        import time

        start = time.time()

        for tt in range(n_tests):
            n_vars = tt + 2

            # Objective function
            h_mat = jnp.eye(n_vars)
            f_vec = -2 * jnp.matmul(h_mat, jnp.array([ii for ii in range(n_vars)]))

            # Inequality Constraints
            g_mat = None
            h_vec = None

            # Equality constraints (random)
            b_vec = jnp.array([ii for ii in range(n_vars)], dtype=jnp.float32)
            u_mat, _ = jnp.linalg.qr(random.normal(KEY, (n_vars, n_vars)))
            v_mat, _ = jnp.linalg.qr(u_mat)
            a_mat = jnp.matmul(u_mat, jnp.diag(b_vec + 1) @ v_mat.T)

            x, status = solver(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)

            # Correct status
            self.assertTrue(status)

            # Solution satisfies equality constraints
            self.assertTrue(
                jnp.sum(abs(jnp.matmul(a_mat, x).flatten() - b_vec)) < self.eps,
                f"Failed Equality constraints: {a_mat} * {x} - {b_vec} = 0",
            )

            #! Difficult to test optimality on equality manifold, so skipping for now
        print(f"test_qp_feasible_equality_only: {time.time() - start}")

    def _test_qp_feasible(self, solver: QpSolverCallable):
        """Tests that the quadratic program solver computes the correct solution for a sequence of
        problems with both inequality and equality constraints."""
        n_tests = 30 if not os.getenv("CBFKIT_TEST_MODE") else 2

        import time

        start = time.time()

        for tt in range(n_tests):
            n_vars = tt + 2

            # Objective function
            h_mat = jnp.eye(n_vars)
            f_vec = -2 * h_mat @ jnp.array([ii for ii in range(n_vars)])

            # Inequality Constraints
            g_mat = jnp.vstack([jnp.eye(n_vars), -jnp.eye(n_vars)])
            h_vec = 100 * jnp.ones((2 * n_vars,))

            # Equality constraints (random)
            b_vec = jnp.array([ii for ii in range(n_vars)], dtype=jnp.float32)
            u_mat, _ = jnp.linalg.qr(random.normal(KEY, (n_vars, n_vars)))
            v_mat, _ = jnp.linalg.qr(u_mat)
            a_mat = u_mat @ jnp.diag(b_vec + 1) @ v_mat.T

            x, status = solver(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)

            # Correct status
            self.assertTrue(status)

            # Solution satisfies equality constraints
            self.assertTrue(
                jnp.sum(abs(jnp.matmul(a_mat, x).flatten() - b_vec)) < self.eps,
                f"Failed Equality constraints: {a_mat} * {x} - {b_vec} = 0",
            )

            # Solution satisfies inequality constraints
            self.assertTrue(
                jnp.sum((jnp.matmul(g_mat, x).flatten() - h_vec) > self.eps) == 0,
                f"Failed Inequality constraints: {g_mat} * {x} - {h_vec} < 0",
            )

        print(f"test_qp_feasible: {time.time() - start}")

    def _test_qp_infeasible(self, solver: QpSolverCallable):
        """Tests that the quadratic program solver correctly identifies that the posed
        inequality/equality constrained problem is infeasible."""
        n_tests = 30 if not os.getenv("CBFKIT_TEST_MODE") else 2

        import time

        start = time.time()

        for tt in range(n_tests):
            n_vars = tt + 2

            # Objective function
            h_mat = jnp.eye(n_vars)
            f_vec = -2 * h_mat @ jnp.array([ii for ii in range(n_vars)])

            # Inequality Constraints
            g_mat = jnp.vstack([jnp.eye(n_vars), -jnp.eye(n_vars)])
            h_vec = -1 * jnp.ones((2 * n_vars,))

            # Equality constraints (random)
            b_vec = jnp.array([ii for ii in range(n_vars)], dtype=jnp.float32)
            u_mat, _ = jnp.linalg.qr(random.normal(KEY, (n_vars, n_vars)))
            v_mat, _ = jnp.linalg.qr(u_mat)
            a_mat = u_mat @ jnp.diag(b_vec + 1) @ v_mat.T

            x, status = solver(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)

            # Correct status
            self.assertFalse(status, f"Failed to report Infeasible! Sol = {x}")

        print(f"test_qp_infeasible: {time.time() - start}")

    def _test_qp_feasible_with_state(self, solver: QpSolverCallable):
        """Tests that the quadratic program solver computes the correct solution for a sequence of
        problems with both inequality and equality constraints, returning state."""
        n_tests = 5
        import time

        start = time.time()

        for tt in range(n_tests):
            n_vars = tt + 2
            h_mat = jnp.eye(n_vars)
            f_vec = -2 * h_mat @ jnp.array([ii for ii in range(n_vars)])
            g_mat = jnp.vstack([jnp.eye(n_vars), -jnp.eye(n_vars)])
            h_vec = 100 * jnp.ones((2 * n_vars,))
            b_vec = jnp.array([ii for ii in range(n_vars)], dtype=jnp.float32)
            u_mat, _ = jnp.linalg.qr(random.normal(KEY, (n_vars, n_vars)))
            v_mat, _ = jnp.linalg.qr(u_mat)
            a_mat = u_mat @ jnp.diag(b_vec + 1) @ v_mat.T

            x, status, params = solver(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec)

            self.assertTrue(status)
            self.assertIsNotNone(params)
            # Verify we can use params as warm start (simple check: no error)
            x2, status2, params2 = solver(
                h_mat, f_vec, g_mat, h_vec, a_mat, b_vec, init_params=params
            )
            self.assertTrue(status2)
            # Solution should be same (within tolerance)
            self.assertTrue(jnp.allclose(x, x2, atol=1e-3, rtol=1e-3))

        print(f"test_qp_feasible_with_state: {time.time() - start}")


if __name__ == "__main__":
    unittest.main()
