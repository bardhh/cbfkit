"""Tests for the PDIPM small-QP solver (replaces coord descent under get_solver('fast'))."""

import jax.numpy as jnp
import pytest


class TestStepToBoundary:
    def test_no_negative_directions_returns_one(self):
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import _step_to_boundary

        s = jnp.array([1.0, 2.0, 3.0])
        ds = jnp.array([0.5, 0.1, 0.0])  # all non-decreasing
        alpha = _step_to_boundary(s, ds)
        assert float(alpha) == pytest.approx(1.0)

    def test_one_negative_caps_step(self):
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import _step_to_boundary

        s = jnp.array([1.0, 1.0])
        ds = jnp.array([-0.5, 0.0])  # s[0] hits 0 at alpha=2.0 → capped at 1.0
        alpha = _step_to_boundary(s, ds)
        assert float(alpha) == pytest.approx(1.0)

    def test_tight_cap(self):
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import _step_to_boundary

        s = jnp.array([1.0, 1.0])
        ds = jnp.array([-2.0, 0.0])  # s[0] hits 0 at alpha=0.5
        alpha = _step_to_boundary(s, ds)
        assert float(alpha) == pytest.approx(0.5)

    def test_jit_compatibility(self):
        import jax
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import _step_to_boundary

        jit_fn = jax.jit(_step_to_boundary)
        alpha = jit_fn(jnp.array([1.0]), jnp.array([-0.5]))
        assert float(alpha) == pytest.approx(1.0)  # 1/0.5 = 2.0, capped at 1.0


class TestNewtonReduced:
    def test_H_is_positive_definite(self):
        """H = P + G^T diag(lam/s) G must be PD when P is PD and s, lam > 0."""
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import _solve_newton_reduced
        from jax import random

        key = random.PRNGKey(0)
        P = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        G = random.normal(key, (5, 3))
        s = jnp.array([0.5, 1.0, 1.5, 2.0, 0.1])
        lam = jnp.array([0.2, 0.8, 1.0, 0.05, 2.0])
        rhs = jnp.ones(3)
        # Build H exactly as the helper would
        D = lam / s
        H = P + G.T @ (D[:, None] * G) + 1e-10 * jnp.eye(3)
        eigs = jnp.linalg.eigvalsh((H + H.T) / 2)
        assert float(eigs[0]) > 0, f"H not PD, min eig = {float(eigs[0])}"

    def test_solves_system(self):
        """_solve_newton_reduced returns dx such that H dx = rhs."""
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import _solve_newton_reduced

        P = jnp.diag(jnp.array([1.0, 1.0]))
        G = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        s = jnp.array([1.0, 1.0, 1.0])
        lam = jnp.array([0.5, 0.5, 0.5])
        rhs = jnp.array([1.0, 2.0])
        dx = _solve_newton_reduced(P, G, s, lam, rhs)
        D = lam / s
        H = P + G.T @ (D[:, None] * G) + 1e-10 * jnp.eye(2)
        residual = H @ dx - rhs
        assert float(jnp.max(jnp.abs(residual))) < 1e-8


class TestPdipmIteration:
    def _make_simple_qp(self):
        """Returns P, q, G, h for: min 0.5 x^T x  s.t. -1 <= x_i <= 1, x in R^2."""
        P = jnp.eye(2)
        q = jnp.array([0.5, -0.5])  # unconstrained opt at [-0.5, 0.5]
        G = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1.0]])
        h = jnp.array([1.0, 1.0, 1.0, 1.0])
        return P, q, G, h

    def test_iteration_keeps_strict_interior(self):
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import _pdipm_iteration

        P, q, G, h = self._make_simple_qp()
        x = jnp.zeros(2)
        s = jnp.ones(4)
        lam = jnp.ones(4)
        x_new, s_new, lam_new = _pdipm_iteration(P, q, G, h, x, s, lam)
        assert float(jnp.min(s_new)) > 0.0, f"slack went non-positive: {s_new}"
        assert float(jnp.min(lam_new)) > 0.0, f"dual went non-positive: {lam_new}"

    def test_iteration_reduces_residual(self):
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import _pdipm_iteration

        P, q, G, h = self._make_simple_qp()
        x = jnp.zeros(2)
        s = jnp.ones(4)
        lam = jnp.ones(4)

        def residual_norm(x_, s_, lam_):
            r_d = P @ x_ + G.T @ lam_ + q
            r_p = G @ x_ + s_ - h
            r_c = (s_ * lam_).sum() / s_.shape[0]
            return float(jnp.linalg.norm(r_d)) + float(jnp.linalg.norm(r_p)) + r_c

        r0 = residual_norm(x, s, lam)
        x1, s1, lam1 = _pdipm_iteration(P, q, G, h, x, s, lam)
        r1 = residual_norm(x1, s1, lam1)
        assert r1 < r0, f"residual did not decrease: {r0} -> {r1}"


class TestSolveQpPdipm:
    def test_unconstrained_minimum(self):
        """min 0.5 x^T x - 2x_0 - 4x_1  => x = [2, 4] (ignoring constraints)."""
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import solve_qp_pdipm

        P = jnp.eye(2)
        q = jnp.array([-2.0, -4.0])
        G = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1.0]])
        h = jnp.array([100.0, 100.0, 100.0, 100.0])  # loose
        x, status, _ = solve_qp_pdipm(P, q, G, h, max_iter=25)
        assert int(status) == 1
        assert jnp.allclose(x, jnp.array([2.0, 4.0]), atol=1e-4), x

    def test_box_constrained_corner(self):
        """min 0.5 x^T x - 10 x_0 - 10 x_1 s.t. |x_i| <= 1 => x = [1, 1]."""
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import solve_qp_pdipm

        P = jnp.eye(2)
        q = jnp.array([-10.0, -10.0])
        G = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1.0]])
        h = jnp.array([1.0, 1.0, 1.0, 1.0])
        x, status, _ = solve_qp_pdipm(P, q, G, h, max_iter=25)
        assert int(status) == 1
        assert jnp.allclose(x, jnp.array([1.0, 1.0]), atol=1e-4), x


class TestPdipmCorrectness:
    def test_cbf_like_constraint(self):
        """Typical CBF-QP: minimize deviation from nominal subject to safety + slack penalty."""
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import solve_qp_pdipm

        P = jnp.diag(jnp.array([1.0, 1.0, 2000.0]))  # 2 controls + 1 slack
        q = jnp.array([-1.0, 0.0, 0.0])  # nominal = [1, 0]
        G = jnp.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [-0.8, -0.2, -1.0],  # CBF constraint with slack
            ]
        )
        h = jnp.array([1.0, 1.0, 1.0, 1.0, 0.1])
        x, status, _ = solve_qp_pdipm(P, q, G, h, max_iter=25)
        assert int(status) == 1
        assert float(jnp.max(G @ x - h)) < 1e-4

    def test_random_pd_qps_match_osqp(self):
        """Random PD QPs: PDIPM objective matches OSQP to within 1% relative error."""
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import solve_qp_pdipm
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="jaxopt")
            from jaxopt import OSQP
        from jax import random

        key = random.PRNGKey(42)
        for trial in range(10):
            key, k1, k2, k3, k4 = random.split(key, 5)
            n = int(random.randint(k1, (), 2, 6))
            m = int(random.randint(k2, (), n + 1, 3 * n))
            P = jnp.diag(jnp.abs(random.normal(k3, (n,))) + 0.1)
            q = random.normal(k4, (n,))
            G = random.normal(k1, (m, n))
            h_vec = jnp.abs(random.normal(k2, (m,))) + 0.5

            x_pdipm, status, _ = solve_qp_pdipm(P, q, G, h_vec, max_iter=25)
            assert int(status) == 1, f"trial {trial}: PDIPM status={int(status)}"

            osqp = OSQP(maxiter=20000, tol=1e-7)
            sol_osqp, _ = osqp.run(params_obj=(P, q), params_ineq=(G, h_vec))

            obj_pdipm = float(0.5 * x_pdipm @ P @ x_pdipm + q @ x_pdipm)
            obj_osqp = float(0.5 * sol_osqp.primal @ P @ sol_osqp.primal + q @ sol_osqp.primal)
            rel_err = abs(obj_pdipm - obj_osqp) / max(abs(obj_osqp), 1e-6)
            assert rel_err < 0.01, f"trial {trial}: pdipm={obj_pdipm} osqp={obj_osqp}"

    def test_constraint_satisfaction_random(self):
        """All PDIPM solutions must satisfy G x <= h within tolerance."""
        from cbfkit.optimization.quadratic_program.qp_solver_pdipm import solve_qp_pdipm
        from jax import random

        key = random.PRNGKey(0)
        for _ in range(10):
            key, k1, k2, k3, k4 = random.split(key, 5)
            n, m = 4, 8
            P = jnp.diag(jnp.abs(random.normal(k1, (n,))) + 0.1)
            q = random.normal(k2, (n,))
            G = random.normal(k3, (m, n))
            h = jnp.abs(random.normal(k4, (m,))) + 0.5
            x, _, _ = solve_qp_pdipm(P, q, G, h, max_iter=25)
            assert float(jnp.max(G @ x - h)) < 1e-3
