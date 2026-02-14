import unittest

try:
    import casadi
except ImportError:
    casadi = None

import jax.numpy as jnp
import jax.random as random
import numpy as np

from cbfkit.controllers.adaptive_cvar_cbf import adaptive_cvar_cbf_controller
from cbfkit.utils.user_types import ControllerData


class MockObstacle:
    def __init__(self):
        self.x_curr = np.array([[2.0], [2.0], [0.0], [0.0]])
        self.velocity_xy = np.zeros((2, 1))
        self.radius = 0.5
        self.noise = [[0.01] * 4, [0.01] * 4]


@unittest.skipIf(casadi is None, "CasADi not installed")
class TestAdaptiveCVaRController(unittest.TestCase):
    def test_controller_factory_and_execution(self):
        # Setup
        dt = 0.1
        A = np.eye(4)
        B = np.zeros((4, 2))
        B[2, 0] = dt
        B[3, 1] = dt

        dyn_model = {
            "A": A,
            "B": B,
            "dt": dt,
            "u_min": -np.ones(2),
            "u_max": np.ones(2),
            "radius": 0.5,
        }

        obstacles = [MockObstacle()]
        params = {"S": 5, "beta": 0.99}

        # Create Controller
        controller = adaptive_cvar_cbf_controller(
            dynamics_model=dyn_model, obstacles=obstacles, params=params
        )

        # Execution inputs
        t = 0.0
        x = jnp.zeros(4)
        u_nom = jnp.zeros(2)
        key = random.PRNGKey(0)
        data = ControllerData()

        # Call
        u, new_data = controller(t, x, u_nom, key, data)

        # Assertions
        self.assertEqual(u.shape, (2,))
        self.assertIsInstance(new_data, ControllerData)
        self.assertIn("prev_solu", new_data.sub_data)

        # Check that optimization returned something reasonable (not all zeros if cost pushes it, but here cost is 0 for u_nom=0)
        # u should be close to 0
        np.testing.assert_allclose(u, np.zeros(2), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
