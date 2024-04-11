"""
#! Docstring
"""

from typing import Tuple
import jax.numpy as jnp
from jax import jit, Array

# from typing import Tuple, Callable
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns


#! TO DO: Nonlinear MPC Controller
def nonlinear_mpc_controller() -> ControllerCallable:
    """Generating function for Nonlinear MPC control law."""

    @jit
    def controller(t: float, x: Array) -> ControllerCallableReturns:
        """Function for computing control input based on Nonlinear MPC."""
        return x, {}

    return controller


#! TO DO: Nonlinear MPC
def nonlinear_mpc():
    @jit
    def solve_mpc(t: float, x: Array) -> Tuple[Array, Array]:
        return jnp.zeros((1,)), jnp.zeros((1,))

    return solve_mpc
