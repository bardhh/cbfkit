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
