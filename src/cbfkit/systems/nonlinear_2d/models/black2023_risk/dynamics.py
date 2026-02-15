from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import DynamicsCallable


def velocity_with_flow(
    r: float,
    sigma: Optional[Array] = None,
) -> Callable[[Array], Tuple[Array, Array, Array]]:
    """Returns a function that represents the equations of motion for a 2D nonlinear system
    with flow dynamics.

    States are the following:
        x: x-coordinate
        y: y-coordinate

    Control inputs are the following:
        u_x: control input in x-direction
        u_y: control input in y-direction

    Dynamics:
        Let c = sqrt(max(x^2 + y^2 - r^2, 0.05))
        dx = c * y + (1/c) * u_x
        dy = c * x + (1/c) * u_y

    Args:
        r (float): parameter affecting the flow field radius
        sigma (Optional, Array): diffusion term in stochastic differential equation

    Returns
    -------
        dynamics (Callable): takes state as input and returns dynamics components
            f, g, and s of the form dx = (f(x) + g(x)u)dt + s(x)dw
    """
    if sigma is not None:
        s = sigma
    else:
        s = jnp.zeros((2, 2))

    @jit
    def equations_of_motion(state: Array) -> Tuple[Array, Array, Array]:
        """Computes the drift vector 'f' and control matrix 'g' based on the given state.

        State consists of x (x-coordinate) and y (y-coordinate).

        Args:
            state (Array): state vector

        Returns
        -------
            f, g, s (Tuple of Arrays): drift vector f, control matrix g, diffusion matrix s
        """
        nonlocal s

        x, y = state
        c = jnp.sqrt(jnp.max(jnp.array([x**2 + y**2 - r**2, 5e-2])))
        f = c * jnp.array([y, x])
        g = 1 / c * jnp.eye(2)

        return f, g, s

    return equations_of_motion
