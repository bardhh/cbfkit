import jax.numpy as jnp
from jax import jit, Array
from typing import Optional, Union
from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns


def reverse_van_der_pol_oscillator(
    epsilon: float,
    sigma: Optional[Union[float, None]] = None,
) -> DynamicsCallable:
    """
    Returns a function that represents the equations of motion for the Van der
    Pol oscillator, and specifically returns them in the form of a drift vector
    'f', control matrix 'g', and diffusion matrix 's' (the argument sigma) based
    on the given state.

    The model characterized by this function may be found in Khalil's Nonlinear
    Systems book.

    States are the following:
        x1: 'position' coordinate
        x2: 'velocity' coordinate

    Control inputs are the following:
        u: 'acceleration'

    Args:
        epsilon (float): parameter in the equations of motion
        sigma (Optional, Array): diffusion term in stochastic differential equation

    Returns:
        dynamics (Callable): takes state as input and returns dynamics components
            f, g, and s of the form dx = (f(x) + g(x)u)dt + s(x)dw

    """
    if sigma is not None:
        s = sigma
    else:
        s = jnp.zeros((2, 2))

    @jit
    def equations_of_motion(state: Array) -> DynamicsCallableReturns:
        """
        Computes the drift vector 'f' and control matrix 'g' based on the given state x,
        which consists of x1 ('position' coordinate) and x2 ('velocity' coordinate).

        Args:
            x (Array): state vector

        Returns:
            f, g, s (Tuple of Arrays): drift vector f, control matrix g, diffusion matrix s

        """
        nonlocal s

        x1, x2 = state
        f = jnp.array([-x2, x1 - epsilon * (1 - x1**2) * x2])
        g = jnp.array(
            [
                [0.0],
                [1 / x2],
            ]
        )

        return f, g, s

    return equations_of_motion
