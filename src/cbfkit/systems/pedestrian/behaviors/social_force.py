"""
Social Force Model (SFM) Policy for Pedestrians.

This policy models pedestrian behavior based on two main forces:
1.  **Desired Force**: Driving the pedestrian towards a specified goal.
2.  **Repulsion Force**: Pushing the pedestrian away from other agents (e.g., other pedestrians, robots).

The policy returns the acceleration input for a single integrator pedestrian model.
"""

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit

# Policy interface: policy(t, pedestrian_state, environment_state) -> control_input (acceleration)
# pedestrian_state: [px, py, vx, vy]
# control_input: [ax, ay]


def policy(
    goal: Array,
    desired_speed: float = 1.2,  # m/s
    relaxation_time: float = 0.5,  # seconds
    repulsion_strength: float = 2.1,  # A_i in Helbing's model (force magnitude)
    repulsion_range: float = 0.08,  # B_i in Helbing's model (interaction range)
    pedestrian_radius: float = 0.3,  # radius for interaction calculation
    agent_radius: float = 0.3,  # default radius for other agents
    noise_intensity: float = 0.0,  # magnitude of random noise
) -> Callable[[float, Array, Dict[str, Array]], Array]:
    """
    Returns a callable policy function for a Social Force Model pedestrian.

    Args:
        goal (Array): The desired destination [gx, gy] for this pedestrian.
        desired_speed (float): The pedestrian's desired speed.
        relaxation_time (float): The time for the pedestrian to adapt to its desired velocity.
        repulsion_strength (float): Magnitude of the repulsive force.
        repulsion_range (float): Range of the repulsive force.
        pedestrian_radius (float): The radius of this pedestrian for interaction calculations.
        agent_radius (float): The default radius for other agents for interaction calculations.
        noise_intensity (float): Standard deviation of additive Gaussian noise.

    Returns:
        Callable[[float, Array, Dict[str, Array]], Array]: A policy function that
            returns the computed acceleration vector [ax, ay].
    """

    @jit
    def social_force_policy(
        t: float,  # Current time
        pedestrian_state: Array,  # Pedestrian state [px, py, vx, vy]
        environment_state: Dict[str, Array],  # Contains 'others_states' etc.
        key: Optional[Array] = None,  # Random key
    ) -> Array:
        """
        Computes the acceleration for the pedestrian based on the Social Force Model.
        """

        px, py, vx, vy = pedestrian_state
        current_pos = jnp.array([px, py])
        current_vel = jnp.array([vx, vy])

        # --- 1. Desired Force towards Goal ---
        desired_direction = goal - current_pos
        dist_to_goal = jnp.linalg.norm(desired_direction)

        # Avoid division by zero if already at goal
        unit_desired_direction = jnp.where(
            dist_to_goal > 1e-6, desired_direction / dist_to_goal, jnp.array([0.0, 0.0])
        )

        desired_velocity = desired_speed * unit_desired_direction

        # Force: m * (v_desired - v_current) / tau (assuming m=1)
        F_desired = (desired_velocity - current_vel) / relaxation_time

        # --- 2. Interaction Forces with Other Agents ---
        F_interaction = jnp.array([0.0, 0.0])

        if "others_states" in environment_state and environment_state["others_states"] is not None:
            others_states = environment_state[
                "others_states"
            ]  # Array of [x,y,vx,vy] or [x,y] for robot

            # vmap over others_states to calculate sum of forces
            def calculate_single_interaction_force(other_state):
                other_pos = other_state[:2]  # Assume first two are positions

                direction_to_self = current_pos - other_pos
                distance = jnp.linalg.norm(direction_to_self)

                # Assume other agent is also roughly 'agent_radius'
                sum_radii = pedestrian_radius + agent_radius

                # Repulsive force only when agents are close
                force_magnitude = jnp.where(
                    distance < sum_radii + repulsion_range,  # Only interact if within range
                    repulsion_strength * jnp.exp((sum_radii - distance) / repulsion_range),
                    0.0,
                )

                # Avoid division by zero
                unit_direction = jnp.where(
                    distance > 1e-6, direction_to_self / distance, jnp.array([0.0, 0.0])
                )

                return force_magnitude * unit_direction

            # Use vmap to apply interaction calculation to all other agents
            # Note: This vmap assumes others_states is 2D array: (num_others, state_dim_other)
            all_interaction_forces = jnp.where(
                others_states.shape[0] > 0,  # Only vmap if there are others
                jax.vmap(calculate_single_interaction_force)(others_states),
                jnp.zeros((1, 2)),  # Placeholder if no others, will be summed to zero
            )

            F_interaction = jnp.sum(all_interaction_forces, axis=0)

        # --- 3. Random Fluctuations ---
        F_fluctuation = jnp.array([0.0, 0.0])
        if key is not None and noise_intensity > 0.0:
            F_fluctuation = jax.random.normal(key, shape=(2,)) * noise_intensity

        # --- Total Acceleration ---
        total_acceleration = F_desired + F_interaction + F_fluctuation

        return total_acceleration

    return social_force_policy
