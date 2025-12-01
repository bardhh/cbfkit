import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns


def double_integrator_dynamics(**kwargs) -> DynamicsCallable:
    """Returns a function that represents the plant model, which computes the drift vector 'f' and
    control matrix 'g' based on the given state.

    States are the following:
        px: x-position (m)
        py: y-position (m)
        vx: x-velocity (m/s)
        vy: y-velocity (m/s)

    Control inputs are the following:
        ax: x-acceleration (m/s^2)
        ay: y-acceleration (m/s^2)

    Args:
        kwargs: keyword arguments

    Returns
    -------
        dynamics (Callable): takes state as input and returns dynamics components
            f, g of the form dx/dt = f(x) + g(x)u
    """

    @jit
    def dynamics(x: Array) -> DynamicsCallableReturns:
        """Computes the drift vector 'f' and control matrix 'g' based on the given state x.

        Args:
            x (Array): state vector

        Returns
        -------
            dynamics (DynamicsCallable): takes state as input and returns dynamics components f, g
        """
        # f(x) = [vx, vy, 0, 0]
        f = jnp.array([x[2], x[3], 0.0, 0.0])

        # g(x) = [[0, 0], [0, 0], [1, 0], [0, 1]]
        g = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        return f, g

    return dynamics
