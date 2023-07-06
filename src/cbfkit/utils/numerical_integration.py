from jax import Array


def forward_euler(x: Array, xdot: Array, dt: float) -> Array:
    """Performs numerical integration on current state (x) and current state
    derivative (xdot) over time interval of length dt according to Forward-Euler
    discretization.

    Arguments:
        x: current state
        xdot: current state derivative
        dt: timestep length (in sec)

    Returns:
        new_state

    """
    return x + xdot * dt
