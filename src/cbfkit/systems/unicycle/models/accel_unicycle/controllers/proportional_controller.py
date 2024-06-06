import jax.numpy as jnp
from jax import jit


def proportional_controller(dynamics, Kp_pos, Kp_theta, desired_state):
    """
    Create a proportional-only controller for the given unicycle dynamics.

    :param dynamics: approximate unicycle dynamics ode
    :param Kp_pos: Position proportional gain.
    :param Kp_theta: Orientation proportional gain.
    :param l: Wheelbase of the unicycle.
    :return: A function that computes control inputs based on the current state and desired state.
    """

    @jit
    def controller(_t, state):
        _, _, v, theta = state
        _, _, _, theta_desired = desired_state

        # Compute the error between the current state and the desired state
        error_pos = jnp.subtract(desired_state[:2], state[:2])
        theta_d = jnp.arctan2(error_pos[1], error_pos[0])
        v_d = jnp.linalg.norm(Kp_pos * error_pos)
        v_d = jnp.minimum(2.0, v_d)
        theta_error = (theta_d - theta) % (2 * jnp.pi)

        unicycle_control_inputs = jnp.array(
            [Kp_pos * (v_d - v), Kp_theta * jnp.minimum(theta_error, 2 * jnp.pi - theta_error)]
        )

        # # Compute control inputs based on proportional gains
        # xdot_d = jnp.multiply(Kp_pos, error_pos)
        # v_d = Kp_pos * (jnp.linalg.norm(xdot_d) - v)
        # omega_d = Kp_theta * error_theta

        # # Convert position and orientation control inputs into unicycle control inputs (v, w)
        # f, g = dynamics(state)
        # unicycle_control_inputs = jnp.linalg.pinv(g) @ jnp.stack(jnp.array([xdot_d[0] - f[0], xdot_d[1] - f[1], v_d, omega_d]))

        # logging data
        data = {}

        return unicycle_control_inputs, data

    return controller
