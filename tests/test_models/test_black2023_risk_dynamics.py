"""Test suite for the black2023_risk dynamics model."""

import unittest

import jax.numpy as jnp

import cbfkit.systems.nonlinear_2d.models.black2023_risk.dynamics as black2023_risk


class TestBlack2023RiskDynamics(unittest.TestCase):
    """Test suite for checking black2023_risk model accuracy.

    Args:
        unittest (unittest.TestCase): TestCase class
    """

    def test_velocity_with_flow(self):
        """Tests that the velocity_with_flow dynamics model matches its implementation."""
        r = 1.0
        dynamics = black2023_risk.velocity_with_flow(r=r)

        # Test at a point outside the circle r=1
        state = jnp.array([2.0, 0.0])  # x=2, y=0
        f, g, s = dynamics(state)

        # Calculate expected values
        # c = sqrt(max(x^2 + y^2 - r^2, 0.05))
        # c = sqrt(max(4 + 0 - 1, 0.05)) = sqrt(3)
        c = jnp.sqrt(3.0)

        # f = c * [y, x] = sqrt(3) * [0, 2] = [0, 2*sqrt(3)]
        f_expected = c * jnp.array([0.0, 2.0])

        # g = (1/c) * I
        g_expected = (1.0 / c) * jnp.eye(2)

        # s should be zeros by default
        s_expected = jnp.zeros((2, 2))

        self.assertTrue(jnp.allclose(f, f_expected), f"Drift vector mismatch: {f} vs {f_expected}")
        self.assertTrue(jnp.allclose(g, g_expected), f"Control matrix mismatch: {g} vs {g_expected}")
        self.assertTrue(jnp.allclose(s, s_expected), f"Diffusion matrix mismatch: {s} vs {s_expected}")

    def test_velocity_with_flow_inside_radius(self):
        """Tests that the velocity_with_flow dynamics correctly clamps c inside the radius."""
        r = 1.0
        dynamics = black2023_risk.velocity_with_flow(r=r)

        # Test at origin (inside r=1)
        state = jnp.array([0.0, 0.0])  # x=0, y=0
        f, g, s = dynamics(state)

        # Calculate expected values
        # c = sqrt(max(0 + 0 - 1, 0.05)) = sqrt(0.05)
        c = jnp.sqrt(0.05)

        # f = c * [y, x] = sqrt(0.05) * [0, 0] = [0, 0]
        f_expected = jnp.zeros(2)

        # g = (1/c) * I
        g_expected = (1.0 / c) * jnp.eye(2)

        self.assertTrue(jnp.allclose(f, f_expected), f"Drift vector mismatch: {f} vs {f_expected}")
        self.assertTrue(jnp.allclose(g, g_expected), f"Control matrix mismatch: {g} vs {g_expected}")


if __name__ == "__main__":
    unittest.main()
