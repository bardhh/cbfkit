
import unittest
import jax.numpy as jnp
from cbfkit.systems.fixed_wing_uav.models.beard2014_kinematic.dynamics import plant

class TestFixedWingDynamics(unittest.TestCase):
    def test_state_ordering(self):
        """
        Verify that the state vector ordering is [x, y, z, v, psi, gamma].
        psi (yaw) should be at index 4.
        gamma (flight path angle) should be at index 5.
        """
        dynamics = plant()

        # State: x=0, y=0, z=0, v=10, psi=pi/2 (90 deg), gamma=0
        # If code matches expected [..., v, psi, gamma]:
        #   psi = pi/2, gamma = 0
        #   dx = v cos(psi) cos(gamma) = 10 * 0 * 1 = 0
        #   dy = v sin(psi) cos(gamma) = 10 * 1 * 1 = 10
        #   dz = v sin(gamma) = 10 * 0 = 0

        state = jnp.array([0.0, 0.0, 0.0, 10.0, jnp.pi/2, 0.0])

        f, g = dynamics(state)

        # Check motion is in Y direction (index 1)
        self.assertTrue(jnp.isclose(f[1], 10.0), f"Expected dy/dt=10, got {f[1]}")
        self.assertTrue(jnp.isclose(f[2], 0.0), f"Expected dz/dt=0, got {f[2]}")

        # Confirm that index 4 is indeed psi.
        # If index 4 was gamma, then gamma=pi/2, psi=0
        # dx = 10 * cos(0) * cos(pi/2) = 0
        # dy = 10 * sin(0) * cos(pi/2) = 0
        # dz = 10 * sin(pi/2) = 10
        # So f[2] would be 10. Since we asserted f[2] is 0, index 4 is NOT gamma.

    def test_input_mapping(self):
        """
        Verify input mapping to state derivatives.
        u = [a, omega, tan(phi)]
        omega (input 1) affects gamma (state 5).
        tan(phi) (input 2) affects psi (state 4).
        """
        dynamics = plant()
        state = jnp.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
        f, g = dynamics(state)

        # g matrix shape: (6, 3)
        # Row 4 (index 4, psi) should be affected by input 2 (index 2)
        # dpsi/dt = (g_accel / v) * u[2]
        expected_val = 9.81 / 10.0
        self.assertTrue(jnp.isclose(g[4, 2], expected_val), f"Expected g[4,2]={expected_val}, got {g[4,2]}")

        # Row 5 (index 5, gamma) should be affected by input 1 (index 1)
        # dgamma/dt = u[1]
        self.assertTrue(jnp.isclose(g[5, 1], 1.0), f"Expected g[5,1]=1.0, got {g[5,1]}")

if __name__ == "__main__":
    unittest.main()
