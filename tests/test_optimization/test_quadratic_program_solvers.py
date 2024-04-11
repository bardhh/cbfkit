"""
Test Module for cbfkit.optimization.quadratic_program solvers.
=========================

This module contains unit tests for functionalities in 'quadratic_program'
from 'cbfkit.optimization'.

Tests
-----
The following test that the given solver correctly computes the solution,
or lack thereof, of the posed optimization problem:
- test_qp_cvxopt_feasible_inequality_only: feasible QP with inequality constraints
- test_qp_jaxopt_feasible_inequality_only: feasible QP with inequality constraints
- test_qp_casadi_feasible_inequality_only: feasible QP with inequality constraints
- test_qp_cvxopt_feasible_equality_only: feasible QP with equality constraints
- test_qp_jaxopt_feasible_iquality_only: feasible QP with equality constraints
- test_qp_casadi_feasible_iquality_only: feasible QP with equality constraints
- test_qp_cvxopt_feasible: feasible QP with inequality and equality constraints
- test_qp_jaxopt_feasible: feasible QP with inequality and equality constraints
- test_qp_casadi_feasible: feasible QP with inequality and equality constraints
- test_qp_cvxopt_infeasible: infeasible QP with inequality and equality constraints
- test_qp_jaxopt_infeasible: infeasible QP with inequality and equality constraints
- test_qp_casadi_infeasible: infeasible QP with inequality and equality constraints

Setup
-----
- No set up required

Examples
--------
To run all tests in this module (from the root of the repository):
    $ python -m unittest tests.test_optimization.test_quadratic_program_solvers
"""

import unittest
import jax.numpy as jnp
from jax import random
import cbfkit.optimization.quadratic_program.qp_solver_cvxopt as qp_cvxopt
import cbfkit.optimization.quadratic_program.qp_solver_jaxopt as qp_jaxopt
import cbfkit.optimization.quadratic_program.qp_solver_casadi as qp_casadi
from cbfkit.utils.user_types import QpSolverCallable

KEY = random.PRNGKey(0)


class TestQuadraticProgramSolvers(unittest.TestCase):
    """Takes care of unit tests intended to verify the intended performance
    of quadratic program solvers.

    """

    eps = 1e-2

    # def test_qp_cvxopt_feasible_inequality_only(self):
    #     """Tests that the CVXOPT-based quadratic program solver computes
    #     the correct solution for a sequence of problems with inequality constraints only."""
    #     self._test_qp_feasible_inequality_only(qp_cvxopt.solve)

    def test_qp_jaxopt_feasible_inequality_only(self):
        """Tests that the JAXOPT-based quadratic program solver computes
        the correct solution for a sequence of problems with inequality constraints only."""
        self._test_qp_feasible_inequality_only(qp_jaxopt.solve)

    # def test_qp_casadi_feasible_inequality_only(self):
    #     """Tests that the Casadi-based quadratic program solver computes
    #     the correct solution for a sequence of problems with inequality constraints only."""
    #     self._test_qp_feasible_inequality_only(qp_casadi.solve)

    # def test_qp_cvxopt_feasible_equality_only(self):
    #     """Tests that the CVXOPT-based quadratic program solver computes the correct solution
    #     for a sequence of problems with equality constraints only."""
    #     self._test_qp_feasible_equality_only(qp_cvxopt.solve)

    def test_qp_jaxopt_feasible_equality_only(self):
        """Tests that the JAXOPT-based quadratic program solver computes the correct solution
        for a sequence of problems with equality constraints only."""
        self._test_qp_feasible_equality_only(qp_jaxopt.solve)

    # def test_qp_casadi_feasible_equality_only(self):
    #     """Tests that the Casadi-based quadratic program solver computes the correct solution
    #     for a sequence of problems with equality constraints only."""
    #     self._test_qp_feasible_equality_only(qp_casadi.solve)

    # def test_qp_cvxopt_feasible(self):
    #     """Tests that the CVXOPT-based quadratic program solver computes the correct solution
    #     for a sequence of problems with both inequality and equality constraints."""
    #     self._test_qp_feasible(qp_cvxopt.solve)

    def test_qp_jaxopt_feasible(self):
        """Tests that the JAXOPT-based quadratic program solver computes the correct solution
        for a sequence of problems with both inequality and equality constraints."""
        self._test_qp_feasible(qp_jaxopt.solve)

    # def test_qp_casadi_feasible(self):
    #     """Tests that the Casadi-based quadratic program solver computes the correct solution
    #     for a sequence of problems with both inequality and equality constraints."""
    #     self._test_qp_feasible(qp_casadi.solve)

    # def test_qp_cvxopt_infeasible(self):
    #     """Tests that the CVXOPT-based quadratic program solver correctly identifies that
    #     the posed inequality/equality constrained problem is infeasible."""
    #     self._test_qp_infeasible(qp_cvxopt.solve)

    def test_qp_jaxopt_infeasible(self):
        """Tests that the JAXOPT-based quadratic program solver correctly identifies that
        the posed inequality/equality constrained problem is infeasible."""
        self._test_qp_infeasible(qp_jaxopt.solve)

    # def test_qp_casadi_infeasible(self):
    #     """Tests that the Casadi-based quadratic program solver correctly identifies that
    #     the posed inequality/equality constrained problem is infeasible."""
    #     self._test_qp_infeasible(qp_casadi.solve)

    def _test_qp_feasible_inequality_only(self, solver: QpSolverCallable):
        """Tests that the quadratic program solver computes the correct solution
        for a sequence of problems with inequality constraints only."""
        n_tests = 30

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
        """Tests that the quadratic program solver computes the correct solution
        for a sequence of problems with equality constraints only."""
        n_tests = 30

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
        """Tests that the quadratic program solver computes the correct solution
        for a sequence of problems with both inequality and equality constraints."""
        n_tests = 30

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
        """Tests that the quadratic program solver correctly identifies that
        the posed inequality/equality constrained problem is infeasible."""
        n_tests = 30

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


if __name__ == "__main__":
    unittest.main()
