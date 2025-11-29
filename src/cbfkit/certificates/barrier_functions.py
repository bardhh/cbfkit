from typing import Tuple

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit


@jit
def ellipsoidal_barrier(
    state_and_time: Array,
    obstacle_state: Array,
    ellipsoid_axes: Array,
    system_position_indices: Tuple[int, ...] = (0, 1),
    obstacle_position_indices: Tuple[int, ...] = (0, 1),
    ellipsoid_axis_indices: Tuple[int, ...] = (0, 1),
) -> Array:
    """
    Computes the ellipsoidal barrier function value.
    h(x) = ((x - x_o) / a_1)^2 + ((y - y_o) / a_2)^2 + ... - 1

    Args:
        state_and_time (Array): The system state vector concatenated with time.
        obstacle_state (Array): The obstacle state vector.
        ellipsoid_axes (Array): The ellipsoid semi-axes vector.
        system_position_indices (Tuple[int, ...]): Indices of position in state_and_time.
        obstacle_position_indices (Tuple[int, ...]): Indices of position in obstacle_state.
        ellipsoid_axis_indices (Tuple[int, ...]): Indices of axes in ellipsoid_axes.

    Returns:
        Array: The barrier function value.
    """
    # Extract position components
    pos = state_and_time[jnp.array(system_position_indices)]
    obs_pos = obstacle_state[jnp.array(obstacle_position_indices)]
    axes = ellipsoid_axes[jnp.array(ellipsoid_axis_indices)]

    # Compute barrier value
    # ((x - xo) / a)^2 - 1
    normalized_diff = (pos - obs_pos) / axes
    return jnp.sum(normalized_diff**2) - 1.0


def ellipsoidal_barrier_factory(
    system_position_indices: Tuple[int, ...] = (0, 1),
    obstacle_position_indices: Tuple[int, ...] = (0, 1),
    ellipsoid_axis_indices: Tuple[int, ...] = (0, 1),
):
    """
    Returns a set of functions (cbf, cbf_grad, cbf_hess) for an ellipsoidal barrier,
    configured for specific state indices.

    The returned functions satisfy the signature expected by certificate_package:
    func(obstacle, ellipsoid) -> func(state_and_time) -> value
    """

    def cbf(obstacle: Array, ellipsoid: Array):
        @jit
        def func(state_and_time: Array) -> Array:
            return ellipsoidal_barrier(
                state_and_time,
                obstacle,
                ellipsoid,
                system_position_indices,
                obstacle_position_indices,
                ellipsoid_axis_indices,
            )

        return func

    def cbf_grad(obstacle: Array, ellipsoid: Array):
        # Create a specialized version of the barrier for differentiation
        def partial_barrier(s):
            return ellipsoidal_barrier(
                s,
                obstacle,
                ellipsoid,
                system_position_indices,
                obstacle_position_indices,
                ellipsoid_axis_indices,
            )

        jacobian = jacfwd(partial_barrier)

        @jit
        def func(state_and_time: Array) -> Array:
            return jacobian(state_and_time)

        return func

    def cbf_hess(obstacle: Array, ellipsoid: Array):
        def partial_barrier(s):
            return ellipsoidal_barrier(
                s,
                obstacle,
                ellipsoid,
                system_position_indices,
                obstacle_position_indices,
                ellipsoid_axis_indices,
            )

        hessian = jacrev(jacfwd(partial_barrier))

        @jit
        def func(state_and_time: Array) -> Array:
            return hessian(state_and_time)

        return func

    return cbf, cbf_grad, cbf_hess
