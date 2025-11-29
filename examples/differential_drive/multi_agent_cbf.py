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

"""
Multi-agent CBF controller example for differential drive robots.

This example uses a centralized formulation where the state of all agents is concatenated
into a single system state. This allows for efficient JIT compilation and handling of
inter-agent constraints without re-instantiating controllers at every step.
"""

from jax import jacfwd, jit, vmap
from jax.scipy.linalg import block_diag

import cbfkit.simulation.simulator as sim

# CBFKit imports
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor


class CentralizedMultiAgentController:
    """Centralized Multi-agent CBF controller."""

    def __init__(self, num_agents=2, num_humans=3, num_obstacles=2):
        self.num_agents = num_agents
        self.num_humans = num_humans
        self.num_obstacles = num_obstacles
        self.state_dim_per_agent = 4
        self.control_dim_per_agent = 2
        self.total_state_dim = num_agents * self.state_dim_per_agent
        self.total_control_dim = num_agents * self.control_dim_per_agent

        # Environment
        self.humans = [jnp.array([2.0 + j, 3.0 + j * 0.5, 0.0]) for j in range(num_humans)]
        self.obstacles = [jnp.array([4.0 + j * 2, 2.0, 0.0]) for j in range(num_obstacles)]

        # Parameters
        self.control_bound = 10.0
        self.d_min_human = 0.8
        self.d_min_obstacle = 0.8
        self.d_min_agent = 0.6

        # Single agent dynamics model (for reuse)
        self.single_plant = unicycle.plant(l=1.0)
        self.single_plant.a_max = self.control_bound
        self.single_plant.omega_max = self.control_bound

    def get_centralized_dynamics(self):
        """Constructs the dynamics function for the full system state."""

        def dynamics(x):
            # x shape: (4 * N,)
            x_reshaped = x.reshape(self.num_agents, self.state_dim_per_agent)

            # Compute f, g for each agent
            # single_plant returns f(4,), g(4,2)
            def get_fg(xi):
                return self.single_plant(xi)

            fs, gs = vmap(get_fg)(x_reshaped)

            # Flatten f: (N, 4) -> (4N,)
            f_total = fs.flatten()

            # Block diagonal g: (N, 4, 2) -> (4N, 2N)
            # jax.scipy.linalg.block_diag requires unpacking
            g_total = block_diag(*gs)

            return f_total, g_total

        return dynamics

    def get_centralized_nominal_controller(self, goals):
        """Constructs the nominal controller for the full system."""
        # We need to bind the goals

        def nom_controller(t, x):
            u_noms = []
            for i in range(self.num_agents):
                # Extract agent state
                xi = x[i * 4 : (i + 1) * 4]
                goal = goals[i]

                # Use proportional controller logic directly
                # Copied/Adapted from unicycle.controllers.proportional_controller to avoid JIT nesting issues
                # or simply call it if it's pure function.

                # Logic:
                # pos_error = goal[:2] - xi[:2]
                # theta_d = atan2(ey, ex)
                # v_d = kp * dist
                # omega = ktheta * (theta_d - theta)

                k_pos = 1.5
                k_theta = 1.5

                error_pos = goal[:2] - xi[:2]
                dist = jnp.linalg.norm(error_pos)
                theta_d = jnp.arctan2(error_pos[1], error_pos[0])
                theta_error = theta_d - xi[3]
                theta_error = (theta_error + jnp.pi) % (2 * jnp.pi) - jnp.pi

                v_cmd = k_pos * dist
                v_cmd = jnp.minimum(2.0, v_cmd)  # Saturate

                accel = k_pos * (v_cmd - xi[2])
                omega = k_theta * theta_error

                u_noms.append(jnp.array([accel, omega]))

            return jnp.concatenate(u_noms)

        return nom_controller

    def get_barriers(self, dynamics_func):
        """Constructs all barrier functions for the centralized system."""
        barriers = []

        # 1. Static Obstacles & Humans
        # For each agent i
        for i in range(self.num_agents):
            idx_x = i * 4
            idx_y = i * 4 + 1

            # Obstacles
            for obs in self.obstacles:

                def h_obs(x):
                    dist_sq = (x[idx_x] - obs[0]) ** 2 + (x[idx_y] - obs[1]) ** 2
                    return dist_sq - self.d_min_obstacle**2

                barriers.append(
                    rectify_relative_degree(
                        h_obs, dynamics_func, self.total_state_dim, form="exponential"
                    )(zeroing_barriers.linear_class_k(5.0))
                )

            # Humans
            for human in self.humans:

                def h_hum(x):
                    dist_sq = (x[idx_x] - human[0]) ** 2 + (x[idx_y] - human[1]) ** 2
                    return dist_sq - self.d_min_human**2

                barriers.append(
                    rectify_relative_degree(
                        h_hum, dynamics_func, self.total_state_dim, form="exponential"
                    )(zeroing_barriers.linear_class_k(5.0))
                )

        # 2. Inter-Agent Collisions
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                idx_xi = i * 4
                idx_yi = i * 4 + 1
                idx_xj = j * 4
                idx_yj = j * 4 + 1

                def h_agent(x):
                    dist_sq = (x[idx_xi] - x[idx_xj]) ** 2 + (x[idx_yi] - x[idx_yj]) ** 2
                    return dist_sq - self.d_min_agent**2

                barriers.append(
                    rectify_relative_degree(
                        h_agent, dynamics_func, self.total_state_dim, form="exponential"
                    )(zeroing_barriers.linear_class_k(5.0))
                )

        return concatenate_certificates(*barriers)

    def create_controller(self, goals):
        """Creates the full CBF controller."""
        dynamics = self.get_centralized_dynamics()
        nominal_controller = self.get_centralized_nominal_controller(goals)
        barrier_package = self.get_barriers(dynamics)

        # Wrap nominal controller to match signature (t, x, key, data) -> (u, data)
        def nom_controller_wrapped(t, x, key, data):
            return nominal_controller(t, x), {}

        # Create CBF controller
        # Note: we pass the wrapped nominal controller to the simulation,
        # but we can pass the raw function to cbf_controller's nominal_input if we want
        # the QP to use it. However, cbf_controller signature for nominal_input is
        # actually just the vector u_nom computed outside usually.
        # cbf_controller takes `nominal_input` as a FUNCTION if provided, or uses the one passed in loop.

        # Actually, cbf_controller generator takes `nominal_input` (callable) as an option,
        # but in the loop `controller(t, x, u_nom)` is called.
        # We will let the simulator compute u_nom using `nominal_controller` and pass it to CBF.

        cbf = cbf_controller(
            control_limits=jnp.tile(
                jnp.array([self.control_bound, self.control_bound]), self.num_agents
            ),
            dynamics_func=dynamics,
            barriers=barrier_package,
        )

        return dynamics, nom_controller_wrapped, cbf


def run_simulation():
    """Run centralized multi-agent simulation."""

    print("Multi-Agent Centralized CBF Demo")
    print("=" * 35)

    # Setup
    num_agents = 2
    controller_factory = CentralizedMultiAgentController(
        num_agents=num_agents, num_humans=3, num_obstacles=2
    )

    # Initial Conditions
    # Agent 1: (0, 0), Agent 2: (0, 2)
    initial_states_list = [
        jnp.array([0.0, 0.0, 0.0, 0.0]),
        jnp.array([0.0, 2.0, 0.0, 0.0]),
    ]
    x0 = jnp.concatenate(initial_states_list)

    # Goals
    # Agent 1 -> (8, 4), Agent 2 -> (8, 2)
    goals = [
        jnp.array([8.0, 4.0, 0.0, 0.0]),
        jnp.array([8.0, 2.0, 0.0, 0.0]),
    ]

    # Create System
    dynamics, nominal_controller, safety_controller = controller_factory.create_controller(goals)

    # Simulation
    tf = 8.0
    dt = 0.05

    print("Running CBFKit simulation (Centralized)...")

    x, u, z, p, c_keys, c_values, p_keys, p_values = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=int(tf / dt),
        dynamics=dynamics,
        integrator=integrator,
        nominal_controller=nominal_controller,
        controller=safety_controller,
        sensor=sensor,
        estimator=estimator,
        filepath="examples/differential_drive/results/multi_agent_cbf_results",
        verbose=True,
    )

    return x, u, controller_factory, goals


def plot_results(x, u, factory, goals):
    """Plot results separating the centralized state back into agents."""
    num_agents = factory.num_agents

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Trajectories
    ax = axes[0, 0]
    colors = ["b", "r", "g", "c"]

    for i in range(num_agents):
        # Extract agent state: x[:, i*4 : i*4+2]
        idx_x = i * 4
        idx_y = i * 4 + 1
        ax.plot(x[:, idx_x], x[:, idx_y], "-", linewidth=2, color=colors[i], label=f"Agent {i+1}")
        ax.plot(x[0, idx_x], x[0, idx_y], "o", color=colors[i])
        ax.plot(goals[i][0], goals[i][1], "*", markersize=12, color=colors[i])

    # Obstacles
    for obs in factory.obstacles:
        circle = plt.Circle(
            (obs[0], obs[1]), factory.d_min_obstacle, fill=True, color="k", alpha=0.3
        )
        ax.add_patch(circle)

    # Humans
    for hum in factory.humans:
        circle = plt.Circle(
            (hum[0], hum[1]), factory.d_min_human, fill=True, color="orange", alpha=0.3
        )
        ax.add_patch(circle)

    ax.set_title("Trajectories")
    ax.legend()
    ax.axis("equal")
    ax.grid(True)

    # 2. Velocities
    ax = axes[0, 1]
    for i in range(num_agents):
        idx_v = i * 4 + 2
        ax.plot(x[:, idx_v], color=colors[i], label=f"Agent {i+1}")
    ax.set_title("Velocities")
    ax.legend()
    ax.grid(True)

    # 3. Controls
    ax = axes[1, 0]
    for i in range(num_agents):
        idx_u1 = i * 2
        idx_u2 = i * 2 + 1
        ax.plot(u[:, idx_u1], color=colors[i], linestyle="-", label=f"A{i+1} Accel")
        ax.plot(u[:, idx_u2], color=colors[i], linestyle="--", label=f"A{i+1} Omega")
    ax.set_title("Controls")
    ax.legend()
    ax.grid(True)

    # 4. Inter-agent Distances
    ax = axes[1, 1]
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            # dist(pos_i, pos_j)
            pos_i = x[:, i * 4 : i * 4 + 2]
            pos_j = x[:, j * 4 : j * 4 + 2]
            dists = jnp.linalg.norm(pos_i - pos_j, axis=1)
            ax.plot(dists, label=f"A{i+1}-A{j+1}")

    ax.axhline(factory.d_min_agent, color="r", linestyle="--", label="Min Dist")
    ax.set_title("Inter-Agent Distances")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("examples/differential_drive/results/multi_agent_cbf_centralized.png")
    print("Saved plot to examples/differential_drive/results/multi_agent_cbf_centralized.png")


def animate_results(x, u, factory, goals):
    """Create animated visualization of the multi-agent simulation."""
    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    print("Creating animation...")

    num_agents = factory.num_agents

    # Setup figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Determine bounds
    all_x = []
    all_y = []
    for i in range(num_agents):
        all_x.extend(x[:, i * 4])
        all_y.extend(x[:, i * 4 + 1])
        all_x.append(goals[i][0])
        all_y.append(goals[i][1])

    for obs in factory.obstacles:
        all_x.append(obs[0])
        all_y.append(obs[1])

    margin = 1.0
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Multi-Agent CBF Animation")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Static Elements
    # Obstacles
    for obs in factory.obstacles:
        circle = Circle((obs[0], obs[1]), factory.d_min_obstacle, fill=True, color="k", alpha=0.3)
        ax.add_patch(circle)

    # Humans
    for hum in factory.humans:
        circle = Circle((hum[0], hum[1]), factory.d_min_human, fill=True, color="orange", alpha=0.3)
        ax.add_patch(circle)

    # Goals
    colors = ["b", "r", "g", "c"]
    for i in range(num_agents):
        ax.plot(
            goals[i][0],
            goals[i][1],
            "*",
            markersize=15,
            color=colors[i % len(colors)],
            markeredgecolor="black",
        )

    # Dynamic Elements
    agent_circles = []
    trails = []

    for i in range(num_agents):
        color = colors[i % len(colors)]
        # Agent body
        circle = Circle((0, 0), 0.3, fill=True, color=color, alpha=0.8, zorder=5)
        ax.add_patch(circle)
        agent_circles.append(circle)

        # Trail
        (trail,) = ax.plot([], [], "-", color=color, alpha=0.5, linewidth=2)
        trails.append(trail)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def animate(frame):
        # Time
        t = frame * 0.05  # dt = 0.05
        time_text.set_text(f"Time: {t:.2f}s")

        for i in range(num_agents):
            idx_x = i * 4
            idx_y = i * 4 + 1

            # Update position
            pos_x = x[frame, idx_x]
            pos_y = x[frame, idx_y]
            agent_circles[i].center = (pos_x, pos_y)

            # Update trail
            trails[i].set_data(x[: frame + 1, idx_x], x[: frame + 1, idx_y])

        return agent_circles + trails + [time_text]

    anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=50, blit=True)

    try:
        filepath = "examples/differential_drive/results/multi_agent_cbf.mp4"
        anim.save(filepath, writer="ffmpeg", fps=20)
        print(f"Saved animation to {filepath}")
    except Exception as e:
        print(f"Could not save MP4: {e}")
        try:
            filepath = "examples/differential_drive/results/multi_agent_cbf.gif"
            anim.save(filepath, writer="pillow", fps=20)
            print(f"Saved animation to {filepath}")
        except Exception as e2:
            print(f"Could not save GIF either: {e2}")

    plt.close()


if __name__ == "__main__":
    x, u, factory, goals = run_simulation()
    plot_results(x, u, factory, goals)
    animate_results(x, u, factory, goals)
