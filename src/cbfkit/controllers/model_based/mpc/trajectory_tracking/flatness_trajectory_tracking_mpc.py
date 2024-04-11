"""
#! docstring
"""

import jax.numpy as jnp
from jax import jit, Array, vmap
from typing import Tuple, Callable
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns

from .linear_trajectory_tracking_mpc import linear_mpc_controller, linear_mpc


def flatness_based_mpc(
    flat_to_actual_control_mapping: Callable[[Array, Array, Array], Array],
    flat_to_actual_state_mapping: Callable[[Array, Array, Array], Array],
    actual_to_flat_state_mapping: Callable[[Array, Array, Array], Array],
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    waypoints: Tuple[Array, Array],
    horizon: float,
    dt: float,
):
    """MPC for Differentially Flat System.

    Args:
        Something

    Returns:
        Something
    """

    mpc = linear_mpc(A, B, Q, R, waypoints, horizon, dt)

    def flat_to_actual_control_mapping_vmap(x: Array):
        return vmap(lambda xb, ub: flat_to_actual_control_mapping(x, xb, ub))

    def solve_mpc(t: float, x: Array) -> Tuple[Array, Array]:
        """ """
        ubar, xbar = mpc(t, actual_to_flat_state_mapping(x))
        u_mapped = flat_to_actual_control_mapping_vmap(x)(xbar.T[:-1], ubar.T)
        x_mapped = flat_to_actual_state_mapping(x, xbar, ubar)

        return x_mapped, u_mapped

    return solve_mpc


def flatness_based_mpc_controller(
    input_constraints: Array,
    flat_to_actual_control_mapping: Callable[[Array, Array], Array],
    actual_to_flat_state_mapping: Callable[[Array, Array, Array], Array],
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    waypoints: Tuple[Array, Array],
    horizon: float,
    dt: float,
) -> ControllerCallable:
    """Wrapper for Differential Flatness based MPC Control law."""
    mpc_controller = linear_mpc_controller(A, B, Q, R, waypoints, horizon, dt)

    @jit
    def controller(t: float, x: Array) -> ControllerCallableReturns:
        """Method for Differential Flatness based MPC Control law."""
        _, data = mpc_controller(t, actual_to_flat_state_mapping(x))
        zn_full = data["xn_full"]
        un_full = data["un_full"]

        u_raw = flat_to_actual_control_mapping(x, zn_full[:, 0], un_full[:, 0])
        u_clipped = jnp.clip(u_raw, -input_constraints, input_constraints)

        return u_clipped, data

    return controller
