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
        _, _, theta = state
        _, _, theta_desired = desired_state

        # Compute the error between the current state and the desired state
        error_pos = jnp.subtract(desired_state[:2], state[:2])
        error_theta = theta_desired - theta

        # Compute control inputs based on proportional gains
        control_inputs_pos = jnp.multiply(Kp_pos, error_pos)
        control_input_theta = Kp_theta * error_theta

        # Convert position and orientation control inputs into unicycle control inputs (v, w)
        g = dynamics(state)[1]
        unicycle_control_inputs = jnp.linalg.pinv(g) @ jnp.hstack(
            (control_inputs_pos, control_input_theta)
        )

        # logging data
        data = {}

        return unicycle_control_inputs, data

    return controller
