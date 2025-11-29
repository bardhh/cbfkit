import jax.numpy as jnp
from jax import jit


def proportional_controller(dynamics, Kp_pos, Kp_theta):
    """Create a proportional-only controller for the given unicycle dynamics.

    :param dynamics: approximate unicycle dynamics ode
    :param Kp_pos: Position proportional gain.
    :param Kp_theta: Orientation proportional gain.
    :param l: Wheelbase of the unicycle.
    :return: A function that computes control inputs based on the current state and desired state.
    """

    @jit
    def controller(_t, state, key, xd):
        _, _, v, theta = state
        _, _, _, theta_desired = xd  # desired_state

        # Compute the error between the current state and the desired state
        error_pos = jnp.subtract(xd[:2], state[:2])
        theta_d = jnp.arctan2(error_pos[1], error_pos[0])

        # Calculate distance to goal
        dist = jnp.linalg.norm(error_pos)

        # Calculate robust heading error in [-pi, pi]
        theta_error = theta_d - theta
        theta_error = (theta_error + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # Desired velocity
        # Saturate max desired velocity at 2.0 (as in original)
        v_d = Kp_pos * dist
        v_d = jnp.minimum(2.0, v_d)

        # Coupling: Slow down if not facing goal to turn in place
        # Use a cosine factor or similar. If error is 90 deg, v_d becomes 0.
        # v_d = v_d * jnp.maximum(0.0, jnp.cos(theta_error))

        # Acceleration control
        # Using Kp_pos as gain for velocity error as well (heuristic from original)
        accel = Kp_pos * (v_d - v)

        # Angular velocity control
        omega = Kp_theta * theta_error

        unicycle_control_inputs = jnp.array([accel, omega])

        # logging data
        data = {}

        return unicycle_control_inputs, data

    return controller
