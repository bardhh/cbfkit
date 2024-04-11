"""
#! docstring

"""
import unittest
import jax.numpy as jnp
from jax import random
import cbfkit.systems as systems

KEY = random.PRNGKey(0)


class TestCbfkitModels(unittest.TestCase):
    """_summary_

    Args:
        unittest (_type_): _description_
    """

    def test_unicycle_model(self):
        """Tests that the unicycle dynamics model is correct according to a
        selection of benchmarks."""
        dynamics = systems.unicycle.models.olfatisaber2002approximate.plant(l=1.0)

        state_1 = jnp.array([0.0, 0.0, 0.0])
        f, g = dynamics(state_1)
        self.assertTrue((f == jnp.zeros((3,))).all())
        self.assertTrue((g == jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])).all())

        state_2 = jnp.array([1.0, 2.0, jnp.pi / 4])
        f, g = dynamics(state_2)
        self.assertTrue((f == jnp.zeros((3,))).all())
        self.assertTrue(
            (
                g
                == jnp.array(
                    [
                        [jnp.sqrt(2) / 2, -jnp.sqrt(2) / 2],
                        [jnp.sqrt(2) / 2, jnp.sqrt(2) / 2],
                        [0.0, 1.0],
                    ]
                )
            ).all()
        )

    def test_quadrotor_model(self):
        """Tests that the quadrotor dynamics model is correct according to a
        selection of benchmarks."""
        m, jx, jy, jz = 1.0, 1.0, 1.0, 1.0
        dynamics = systems.quadrotor_6dof.models.quadrotor_6dof_dynamics(m=m, jx=jx, jy=jy, jz=jz)

        # Dropped vertically from zero vel
        state_1 = jnp.zeros((12,))
        f, g, s = dynamics(state_1)

        f_pred = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        g_pred = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-1 / m, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1 / jx, 0.0, 0.0],
                [0.0, 0.0, 1 / jy, 0.0],
                [0.0, 0.0, 0.0, 1 / jz],
            ]
        )

        self.assertTrue((f == f_pred).all())
        self.assertTrue((g == g_pred).all())
        self.assertTrue((s == jnp.zeros((12, 12))).all())

        # Falling altitude
        state_1 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        f, g, s = dynamics(state_1)

        f_pred = jnp.array([0.0, 0.0, -1.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        g_pred = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-1 / m, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1 / jx, 0.0, 0.0],
                [0.0, 0.0, 1 / jy, 0.0],
                [0.0, 0.0, 0.0, 1 / jz],
            ]
        )

        self.assertTrue((f == f_pred).all())
        self.assertTrue((g == g_pred).all())
        self.assertTrue((s == jnp.zeros((12, 12))).all())

        # Thrust counteracts gravity
        state_1 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        f, g, s = dynamics(state_1)
        control = jnp.array([9.81, 0.0, 0.0, 0.0])
        xdot = f + jnp.matmul(g, control)

        self.assertTrue((xdot == jnp.zeros((12,))).all())


if __name__ == "__main__":
    unittest.main()
