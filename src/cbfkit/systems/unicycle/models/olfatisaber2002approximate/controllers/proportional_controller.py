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
        _, _, final_theta_desired = desired_state

        # Compute the error between the current state and the desired state
        error_pos = jnp.subtract(desired_state[:2], state[:2])

        # Desired orientation is to point towards the goal
        theta_desired = jnp.arctan2(error_pos[1], error_pos[0])

        # When close to the goal, switch to final desired orientation
        dist_to_goal = jnp.linalg.norm(error_pos)
        # Note: this threshold may need tuning
        theta_desired = jnp.where(dist_to_goal > 0.1, theta_desired, final_theta_desired)

        error_theta = theta_desired - theta
        # Normalize angle error to [-pi, pi]
        error_theta = jnp.arctan2(jnp.sin(error_theta), jnp.cos(error_theta))

        # Compute control inputs based on proportional gains
        control_inputs_pos = jnp.multiply(Kp_pos, error_pos)
        control_input_theta = Kp_theta * error_theta

        # Convert position and orientation control inputs into unicycle control inputs (v, w)
        g = dynamics(state)[1]
        unicycle_control_inputs = jnp.linalg.pinv(g) @ jnp.hstack(
            (control_inputs_pos, control_input_theta)
        )

        # Check if controller is within goal tolerance
        data = {"complete": dist_to_goal <= 0.1}

        return unicycle_control_inputs, data

    return controller
