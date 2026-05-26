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
