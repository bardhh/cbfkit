"""
Multi-agent CBF controller example for differential drive robots.

This example follows the structure of past_proj files but uses CBFKit machinery
for multi-agent coordination with collision avoidance between agents, humans, and obstacles.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# CBFKit imports
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller
from cbfkit.controllers.cbf_clf.utils.barrier_conditions import (
    zeroing_barriers,
)
from cbfkit.controllers.cbf_clf.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers.cbf_clf.utils.rectify_relative_degree import (
    rectify_relative_degree,
)


class MultiAgentCBFController:
    """Multi-agent CBF controller following past_proj structure but using CBFKit."""

    def __init__(self, num_agents=2, num_humans=3, num_obstacles=2):
        self.num_agents = num_agents
        self.num_humans = num_humans
        self.num_obstacles = num_obstacles

        # CBF parameters (following past_proj values)
        self.alpha1_human = 1.0 * np.ones(num_humans)
        self.alpha2_human = 2.0 * np.ones(num_humans)
        self.alpha1_obstacle = 2.0 * np.ones(num_obstacles)
        self.alpha2_obstacle = 6.0 * np.ones(num_obstacles)
        self.alpha1_agent = 2.0
        self.alpha2_agent = 5.0

        # Control parameters
        self.control_bound = 1.0
        self.robot_radius = 0.3
        self.d_min_human = 0.8
        self.d_min_obstacle = 0.8
        self.d_min_agent = 0.6

        # Goal parameters
        self.k_x = 1.5
        self.k_v = 3.0
        self.k_omega = 1.5

        # Initialize agents
        self.agents = []
        self.controllers = []
        self.agent_states = []
        self.agent_goals = []

        # Create individual unicycle dynamics for each agent
        for _ in range(num_agents):
            dynamics = unicycle.plant(l=1.0)
            dynamics.a_max = self.control_bound
            dynamics.omega_max = self.control_bound
            dynamics.v_max = 3.0
            dynamics.goal_tol = 0.25
            self.agents.append(dynamics)

    def setup_agents(self, initial_states, goals):
        """Setup agent initial states and goals."""
        self.agent_states = [jnp.array(state) for state in initial_states]
        self.agent_goals = [jnp.array(goal) for goal in goals]

        from jax import random

        # Create controllers for each agent
        for i in range(self.num_agents):
            # Create nominal controller for this agent
            base_nom_controller = unicycle.controllers.proportional_controller(
                dynamics=self.agents[i],
                Kp_pos=self.k_x,
                Kp_theta=self.k_omega,
            )

            # Capture goal in closure and create a callable that matches the expected signature for manual calling
            goal = self.agent_goals[i]

            def make_nom_controller(base_ctrl=base_nom_controller, target=goal):
                def ctrl_func(t, state):
                    key = random.PRNGKey(0)
                    u, _ = base_ctrl(t, state, key, target)
                    return u

                return ctrl_func

            # Store the nominal controller function
            nom_controller = make_nom_controller()
            # We attach it to the controller list or a separate list to use in step()
            # But self.controllers stores the CBF controllers. Let's store nom_controllers separately.
            if not hasattr(self, "nom_controllers"):
                self.nom_controllers = []
            self.nom_controllers.append(nom_controller)

            # Create barriers for this agent
            barriers = self._create_barriers_for_agent(i)

            # Create CBF controller
            # nominal_input is NOT passed here
            controller = cbf_controller(
                control_limits=jnp.array([self.control_bound, self.control_bound]),
                dynamics_func=self.agents[i],
                barriers=barriers,
            )

            self.controllers.append(controller)

    def _create_barriers_for_agent(self, agent_idx):
        """Create barrier functions for a specific agent."""
        barriers = []

        cbf_factory, _, _ = ellipsoidal_barrier_factory(
            system_position_indices=(0, 1),
            obstacle_position_indices=(0, 1),
            ellipsoid_axis_indices=(0, 1),
        )

        # Human avoidance barriers
        for j in range(self.num_humans):
            human_pos = jnp.array([2.0 + j, 3.0 + j * 0.5, 0.0])  # Example human positions
            barrier = rectify_relative_degree(
                function=cbf_factory(human_pos, (self.d_min_human, self.d_min_human)),
                system_dynamics=self.agents[agent_idx],
                state_dim=4,
                form="exponential",
            )(
                certificate_conditions=zeroing_barriers.linear_class_k(
                    self.alpha1_human[j] + self.alpha2_human[j]
                ),
                obstacle=human_pos,
                ellipsoid=(self.d_min_human, self.d_min_human),
            )
            barriers.append(barrier)

        # Obstacle avoidance barriers
        for j in range(self.num_obstacles):
            obstacle_pos = jnp.array([4.0 + j * 2, 2.0, 0.0])  # Example obstacle positions
            barrier = rectify_relative_degree(
                function=cbf_factory(obstacle_pos, (self.d_min_obstacle, self.d_min_obstacle)),
                system_dynamics=self.agents[agent_idx],
                state_dim=4,
                form="exponential",
            )(
                certificate_conditions=zeroing_barriers.linear_class_k(
                    self.alpha1_obstacle[j] + self.alpha2_obstacle[j]
                ),
                obstacle=obstacle_pos,
                ellipsoid=(self.d_min_obstacle, self.d_min_obstacle),
            )
            barriers.append(barrier)

        # Agent-to-agent collision avoidance barriers
        for j in range(self.num_agents):
            if j != agent_idx:
                # Create inter-agent barrier (will be updated with other agent's state)
                other_agent_pos = jnp.array([0.0, 0.0, 0.0])  # Placeholder
                barrier = rectify_relative_degree(
                    function=cbf_factory(other_agent_pos, (self.d_min_agent, self.d_min_agent)),
                    system_dynamics=self.agents[agent_idx],
                    state_dim=4,
                    form="exponential",
                )(
                    certificate_conditions=zeroing_barriers.linear_class_k(
                        self.alpha1_agent + self.alpha2_agent
                    ),
                    obstacle=other_agent_pos,
                    ellipsoid=(self.d_min_agent, self.d_min_agent),
                )
                barriers.append(barrier)

        return concatenate_certificates(*barriers) if barriers else None

    def step(self, dt, human_states=None, human_velocities=None):
        """Execute one simulation step for all agents."""
        if human_states is None:
            human_states = [
                jnp.array([2.0 + i, 3.0 + i * 0.5, 0.0, 0.0]) for i in range(self.num_humans)
            ]
        if human_velocities is None:
            human_velocities = [jnp.array([0.1, 0.0, 0.0, 0.0]) for _ in range(self.num_humans)]

        from jax import random

        from cbfkit.utils.user_types import ControllerData

        # Compute control inputs for all agents
        controls = []
        for i in range(self.num_agents):
            # Update barriers with current positions of other agents
            # This is a simplified approach - in practice, you'd want to update
            # the barrier functions dynamically

            # Compute nominal control
            u_nom = self.nom_controllers[i](0.0, self.agent_states[i])

            # Compute control input - CBF controller takes (t, x, u_nom, key, data)
            # We need to provide a key and a dummy data object
            key = random.PRNGKey(0)
            data = ControllerData()

            control_input, _ = self.controllers[i](
                t=0.0, x=self.agent_states[i], u_nom=u_nom, key=key, data=data
            )
            controls.append(control_input)

        # Update agent states
        new_states = []
        for i in range(self.num_agents):
            # Forward Euler integration using CBFKit dynamics
            f_val, g_val = self.agents[i](self.agent_states[i])
            xdot = f_val + g_val @ controls[i]
            new_state = self.agent_states[i] + xdot * dt
            new_states.append(new_state)

        self.agent_states = new_states
        return self.agent_states, controls

    def run_simulation(self, tf=10.0, dt=0.01, save_results=True):
        """Run multi-agent simulation."""
        print(f"Running multi-agent simulation with {self.num_agents} agents...")

        # Simulation data storage
        t_span = np.arange(0, tf, dt)
        states_history = {i: [self.agent_states[i]] for i in range(self.num_agents)}
        controls_history = {i: [] for i in range(self.num_agents)}

        # Main simulation loop
        for step_idx, t in enumerate(t_span[1:]):
            try:
                states, controls = self.step(dt)

                # Store data
                for i in range(self.num_agents):
                    states_history[i].append(states[i])
                    controls_history[i].append(controls[i])

                # Print progress every 100 steps
                if step_idx % 100 == 0:
                    print(f"Step {step_idx}, t={t:.2f}s")
                    for i in range(self.num_agents):
                        pos = states[i][:2]
                        goal_dist = np.linalg.norm(pos - self.agent_goals[i][:2])
                        print(
                            f"  Agent {i+1}: pos=({pos[0]:.2f}, {pos[1]:.2f}), goal_dist={goal_dist:.2f}"
                        )

                # Check if all agents reached their goals
                all_reached = True
                for i in range(self.num_agents):
                    goal_dist = np.linalg.norm(states[i][:2] - self.agent_goals[i][:2])
                    if goal_dist > 0.5:
                        all_reached = False
                        break

                if all_reached:
                    print(f"All agents reached their goals at t={t:.2f}s")
                    break

            except Exception as e:
                print(f"Error at step {step_idx}: {e}")
                break

        # Convert to numpy arrays
        for i in range(self.num_agents):
            states_history[i] = np.array(states_history[i])
            if controls_history[i]:
                controls_history[i] = np.array(controls_history[i])

        if save_results:
            self.plot_results(states_history, controls_history)

        return states_history, controls_history

    def plot_results(self, states_history, controls_history):
        """Plot simulation results."""
        _, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot trajectories
        ax = axes[0, 0]
        colors = ["b", "r", "g", "c", "m", "y"]

        for i in range(self.num_agents):
            states = states_history[i]
            color = colors[i % len(colors)]

            # Plot trajectory
            ax.plot(states[:, 0], states[:, 1], color=color, linewidth=2, label=f"Agent {i+1}")

            # Plot start and goal
            ax.plot(states[0, 0], states[0, 1], "o", color=color, markersize=8)
            ax.plot(self.agent_goals[i][0], self.agent_goals[i][1], "*", color=color, markersize=12)

        # Plot obstacles
        for j in range(self.num_obstacles):
            obstacle_pos = [4.0 + j * 2, 2.0]
            circle = plt.Circle(
                obstacle_pos, self.d_min_obstacle, fill=False, color="k", linewidth=2
            )
            ax.add_patch(circle)

        # Plot humans
        for j in range(self.num_humans):
            human_pos = [2.0 + j, 3.0 + j * 0.5]
            circle = plt.Circle(
                human_pos, self.d_min_human, fill=False, color="orange", linewidth=2
            )
            ax.add_patch(circle)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Multi-Agent Trajectories")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        # Plot velocities
        ax = axes[0, 1]
        for i in range(self.num_agents):
            states = states_history[i]
            if len(states) > 1:
                velocities = np.array([np.linalg.norm(states[j, 2:4]) for j in range(len(states))])
                ax.plot(velocities, color=colors[i % len(colors)], label=f"Agent {i+1}")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Agent Velocities")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot control inputs
        ax = axes[1, 0]
        for i in range(self.num_agents):
            controls = controls_history[i]
            if len(controls) > 0:
                controls = np.array(controls)
                ax.plot(
                    controls[:, 0],
                    color=colors[i % len(colors)],
                    linestyle="-",
                    label=f"Agent {i+1} u1",
                )
                ax.plot(
                    controls[:, 1],
                    color=colors[i % len(colors)],
                    linestyle="--",
                    label=f"Agent {i+1} u2",
                )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Control Input")
        ax.set_title("Control Inputs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot distances between agents
        ax = axes[1, 1]
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                states_i = states_history[i]
                states_j = states_history[j]
                min_len = min(len(states_i), len(states_j))
                distances = [
                    np.linalg.norm(states_i[k, :2] - states_j[k, :2]) for k in range(min_len)
                ]
                ax.plot(distances, label=f"Agent {i+1} - Agent {j+1}")

        ax.axhline(y=self.d_min_agent, color="r", linestyle="--", label="Min Distance")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Inter-Agent Distances")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("examples/differential_drive/results/multi_agent_cbf_results.png", dpi=150)
        plt.show()


def main():
    """Main function to run the multi-agent CBF example."""
    # Create controller
    controller = MultiAgentCBFController(num_agents=2, num_humans=3, num_obstacles=2)

    # Setup initial states and goals
    initial_states = [
        jnp.array([0.0, 0.0, 0.0, 0.0]),  # Agent 1: [x, y, theta, v]
        jnp.array([0.0, 2.0, 0.0, 0.0]),  # Agent 2: [x, y, theta, v]
    ]

    goals = [
        jnp.array([8.0, 4.0, 0.0, 0.0]),  # Agent 1 goal
        jnp.array([8.0, 2.0, 0.0, 0.0]),  # Agent 2 goal
    ]

    # Setup agents
    controller.setup_agents(initial_states, goals)

    # Run simulation
    states_history, _ = controller.run_simulation(
        tf=5.0,  # Shorter simulation for testing
        dt=0.05,  # Larger timestep for faster simulation
        save_results=True,
    )

    # Performance analysis
    print("\nPerformance Analysis:")
    print("=" * 50)

    for i in range(controller.num_agents):
        final_state = states_history[i][-1]
        goal_error = np.linalg.norm(final_state[:2] - controller.agent_goals[i][:2])
        print(f"Agent {i+1} final goal error: {goal_error:.3f} m")

    # Check minimum distances
    min_distances = []
    for i in range(controller.num_agents):
        for j in range(i + 1, controller.num_agents):
            states_i = states_history[i]
            states_j = states_history[j]
            min_len = min(len(states_i), len(states_j))
            distances = [np.linalg.norm(states_i[k, :2] - states_j[k, :2]) for k in range(min_len)]
            min_dist = min(distances)
            min_distances.append(min_dist)
            print(f"Min distance Agent {i+1} - Agent {j+1}: {min_dist:.3f} m")

    overall_min = min(min_distances) if min_distances else float("inf")
    print(f"Overall minimum inter-agent distance: {overall_min:.3f} m")
    print(f"Safety maintained: {'✅' if overall_min > controller.d_min_agent else '❌'}")


if __name__ == "__main__":
    main()
