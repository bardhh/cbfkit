"""
Single robot CBF controller example using CBFKit's simulation framework.

This example follows the structure of past_proj files but uses CBFKit's built-in
simulation capabilities for a single robot with obstacle avoidance.
"""

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle

import cbfkit.simulation.simulator as sim

# CBFKit imports - using the proper simulation framework
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
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor


def create_robot_with_obstacles():
    """Create a single robot with obstacle avoidance using CBFKit framework."""

    # Robot parameters (following past_proj structure)
    control_bound = 100.0
    d_min_obstacle = 0.8

    # Create unicycle dynamics
    dynamics = unicycle.plant(l=1.0)
    dynamics.a_max = control_bound
    dynamics.omega_max = control_bound
    dynamics.v_max = 3.0
    dynamics.goal_tol = 0.25

    # Define scenario
    init_state = jnp.array([0.0, 0.0, 0.0, 0.0])  # [x, y, v, theta]
    desired_state = jnp.array([5.0, 3.0, 0.0, 0.0])  # [x, y, v, theta]

    # Define obstacles (following past_proj format)
    obstacles = [
        (2.0, 1.0, 0.0),  # Obstacle 1: (x, y, z)
        (3.5, 0.5, 0.0),  # Obstacle 2: (x, y, z)
    ]

    # Create nominal controller with higher gains for better goal reaching
    # proportional_controller returns a function with signature (t, state, key, desired_state)
    nom_controller_func = unicycle.controllers.proportional_controller(
        dynamics=dynamics,
        Kp_pos=3.0,  # Higher gain for faster convergence
        Kp_theta=5.0,
    )

    # Define the nominal controller for sim.execute
    # Signature: (t, x, key, planner_data/None) -> (u, data)
    def nominal_control_law(t, x, key, _):
        return nom_controller_func(t, x, key, desired_state)

    # Create barrier functions for each obstacle
    cbf_factory, _, _ = ellipsoidal_barrier_factory(
        system_position_indices=(0, 1),
        obstacle_position_indices=(0, 1),
        ellipsoid_axis_indices=(0, 1),
    )

    barriers = []
    for i, obs in enumerate(obstacles):
        print(f"Creating barrier for obstacle {i+1} at ({obs[0]}, {obs[1]})")

        barrier = rectify_relative_degree(
            function=cbf_factory(jnp.array(obs), (d_min_obstacle, d_min_obstacle)),
            system_dynamics=dynamics,
            state_dim=4,
            form="exponential",
        )(
            certificate_conditions=zeroing_barriers.linear_class_k(5.0),  # past_proj alpha values
            obstacle=jnp.array(obs),
            ellipsoid=(d_min_obstacle, d_min_obstacle),
        )
        barriers.append(barrier)

    # Package all barriers
    barrier_package = concatenate_certificates(*barriers)

    # Create CBF controller using CBFKit framework
    # Note: nominal_input is NOT passed here; it is handled by the simulator/stepper
    controller = cbf_controller(
        control_limits=jnp.array([control_bound, control_bound]),
        dynamics_func=dynamics,
        barriers=barrier_package,
    )

    return (
        dynamics,
        controller,
        nominal_control_law,
        init_state,
        desired_state,
        obstacles,
        d_min_obstacle,
    )


def run_simulation():
    """Run the single robot simulation using CBFKit."""

    print("Single Robot CBF Navigation Demo")
    print("=" * 35)

    # Setup robot and scenario
    (
        dynamics,
        controller,
        nominal_controller,
        init_state,
        desired_state,
        obstacles,
        d_min_obstacle,
    ) = create_robot_with_obstacles()

    print(f"Initial position: [{init_state[0]:.1f}, {init_state[1]:.1f}]")
    print(f"Goal position: [{desired_state[0]:.1f}, {desired_state[1]:.1f}]")
    print(f"Obstacles: {[(obs[0], obs[1]) for obs in obstacles]}")
    print(f"Safety distance: {d_min_obstacle} m")
    print()

    # Simulation parameters
    tf = 12.0  # More time to reach goal
    dt = 0.1

    print("Running CBFKit simulation...")

    # Use CBFKit's built-in simulation framework
    x, u, z, p, c_keys, c_values, p_keys, p_values = sim.execute(
        x0=init_state,
        dt=dt,
        num_steps=int(tf / dt),
        dynamics=dynamics,
        integrator=integrator,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        filepath="examples/differential_drive/results/single_robot_cbf_results",
        verbose=True,
    )

    return x, u, z, p, c_keys, c_values, desired_state, obstacles, d_min_obstacle


def plot_results(states, controls, desired_state, obstacles, d_min_obstacle):
    """Create comprehensive visualization of the results."""

    # Create figure with subplots
    _, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Trajectory with obstacles
    ax = axes[0, 0]
    ax.plot(states[:, 0], states[:, 1], "b-", linewidth=3, label="Robot trajectory")
    ax.plot(states[0, 0], states[0, 1], "go", markersize=10, label="Start")
    ax.plot(desired_state[0], desired_state[1], "r*", markersize=15, label="Goal")

    # Plot obstacles
    for i, obs in enumerate(obstacles):
        circle = plt.Circle(
            (obs[0], obs[1]),
            d_min_obstacle,
            fill=True,
            color="red",
            alpha=0.3,
            label="Safety zone" if i == 0 else "",
        )
        ax.add_patch(circle)
        ax.plot(obs[0], obs[1], "ks", markersize=8, label="Obstacles" if i == 0 else "")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Robot Navigation with CBF Safety")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Plot 2: Distance to goal over time
    ax = axes[0, 1]
    time_vec = np.arange(len(states)) * 0.1
    goal_distances = [np.linalg.norm(state[:2] - desired_state[:2]) for state in states]
    ax.plot(time_vec, goal_distances, "b-", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance to Goal (m)")
    ax.set_title("Goal Convergence")
    ax.grid(True, alpha=0.3)

    # Plot 3: Control inputs
    ax = axes[1, 0]
    time_vec_ctrl = np.arange(len(controls)) * 0.1
    ax.plot(time_vec_ctrl, controls[:, 0], "r-", linewidth=2, label="Acceleration")
    ax.plot(time_vec_ctrl, controls[:, 1], "b-", linewidth=2, label="Angular velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Commands")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Safety distances to obstacles
    ax = axes[1, 1]
    for i, obs in enumerate(obstacles):
        distances = [np.linalg.norm(state[:2] - np.array(obs[:2])) for state in states]
        ax.plot(time_vec, distances, linewidth=2, label=f"Obstacle {i+1}")

    ax.axhline(y=d_min_obstacle, color="red", linestyle="--", linewidth=2, label="Safety threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance to Obstacles (m)")
    ax.set_title("Safety Monitoring")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = "examples/differential_drive/results/single_robot_cbf_comprehensive.png"
    print(f"Saving comprehensive plot to {filename}")
    plt.savefig(filename, dpi=150)
    plt.close()


def create_animation(states, controls, desired_state, obstacles, d_min_obstacle):
    """Create animated visualization of the single robot simulation."""
    print("Creating animation...")

    # Setup figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Find simulation bounds
    all_x = [state[0] for state in states] + [obs[0] for obs in obstacles] + [desired_state[0]]
    all_y = [state[1] for state in states] + [obs[1] for obs in obstacles] + [desired_state[1]]
    margin = 1.5
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Single Robot CBF Navigation Animation", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Draw static elements
    # Obstacles
    for obs in obstacles:
        circle = Circle(
            (obs[0], obs[1]), d_min_obstacle, fill=True, color="red", alpha=0.3, zorder=1
        )
        ax.add_patch(circle)
        ax.plot(obs[0], obs[1], "ks", markersize=8, zorder=2)

    # Goal
    goal_circle = Circle(
        (desired_state[0], desired_state[1]), 0.15, fill=True, color="green", alpha=0.5, zorder=1
    )
    ax.add_patch(goal_circle)
    ax.plot(
        desired_state[0],
        desired_state[1],
        "*",
        color="green",
        markersize=15,
        zorder=3,
        markeredgecolor="black",
        markeredgewidth=1,
    )

    # Initialize robot elements
    robot_circle = Circle(
        (0, 0), 0.15, fill=True, color="blue", alpha=0.8, zorder=4, edgecolor="black", linewidth=2
    )
    ax.add_patch(robot_circle)

    # Robot heading arrow will be managed by arrow_ref

    # Trail line
    (trail_line,) = ax.plot([], [], "b-", alpha=0.6, linewidth=2, zorder=2)

    # Text elements
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    status_text = ax.text(
        0.02,
        0.90,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    control_text = ax.text(
        0.02,
        0.80,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
    )

    # Use list to store arrow reference (mutable)
    arrow_ref = [None]

    def animate_frame(frame):
        """Animation function called for each frame."""
        if frame >= len(states):
            return [robot_circle, trail_line, time_text, status_text, control_text]

        t = frame * 0.1  # dt = 0.1

        # Update time text
        time_text.set_text(f"Time: {t:.1f}s")

        # Get current state
        state = states[frame]
        x, y, v, theta = state[0], state[1], state[2], state[3]

        # Update robot position
        robot_circle.center = (x, y)

        # Update robot heading arrow
        arrow_length = 0.3
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)

        # Remove old arrow if it exists and add new one
        if arrow_ref[0] is not None:
            arrow_ref[0].remove()
        arrow_ref[0] = Arrow(x, y, dx, dy, width=0.2, color="blue", alpha=0.8, zorder=5)
        ax.add_patch(arrow_ref[0])

        # Update trail
        trail_x = [state[0] for state in states[: frame + 1]]
        trail_y = [state[1] for state in states[: frame + 1]]
        trail_line.set_data(trail_x, trail_y)

        # Update status text
        goal_dist = np.linalg.norm(state[:2] - desired_state[:2])
        status_text.set_text(f"Goal distance: {goal_dist:.2f}m\nVelocity: {v:.2f}m/s")

        # Update control text
        if frame < len(controls):
            control = controls[frame]
            control_text.set_text(f"Acceleration: {control[0]:.2f}\nAngular vel: {control[1]:.2f}")

        # Check obstacle distances for safety warning
        min_obs_dist = float("inf")
        for obs in obstacles:
            dist = np.linalg.norm(state[:2] - np.array(obs[:2]))
            min_obs_dist = min(min_obs_dist, dist)

        # Change robot color based on safety
        if min_obs_dist < d_min_obstacle * 1.2:  # Warning zone
            robot_circle.set_facecolor("orange")
        elif min_obs_dist < d_min_obstacle:  # Danger zone
            robot_circle.set_facecolor("red")
        else:
            robot_circle.set_facecolor("blue")

        return [robot_circle, arrow_ref[0], trail_line, time_text, status_text, control_text]

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate_frame,
        frames=len(states),
        interval=100,
        blit=False,
        repeat=True,
    )

    # Save animation
    try:
        print("Saving animation as MP4...")
        filepath = "examples/differential_drive/results/single_robot_cbf_animation.mp4"
        anim.save(filepath, writer="ffmpeg", fps=10, dpi=100)
        print(f"Animation saved to {filepath}")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Saving as GIF instead...")
        try:
            filepath_gif = "examples/differential_drive/results/single_robot_cbf_animation.gif"
            anim.save(filepath_gif, writer="pillow", fps=5)
            print(f"Animation saved as GIF to {filepath_gif}")
        except Exception as e2:
            print(f"Could not save animation as GIF either: {e2}")

    plt.close()
    return anim


def analyze_performance(
    states, controls, c_keys, c_values, desired_state, obstacles, d_min_obstacle
):
    """Analyze the performance following past_proj style."""

    print("\nPerformance Analysis:")
    print("=" * 25)

    # Goal reaching analysis
    final_state = states[-1]
    goal_error = np.linalg.norm(final_state[:2] - desired_state[:2])
    print(f"Final goal error: {goal_error:.3f} m")
    print(f"Goal reached: {'✅' if goal_error < 0.5 else '❌'}")

    # Control effort analysis (following past_proj metrics)
    control_effort = np.sum(np.linalg.norm(controls, axis=1)) * 0.1  # dt = 0.1
    print(f"Total control effort: {control_effort:.2f}")

    # Safety analysis
    min_distances = []
    for obs in obstacles:
        distances = [np.linalg.norm(state[:2] - np.array(obs[:2])) for state in states]
        min_dist = min(distances)
        min_distances.append(min_dist)
        print(f"Min distance to obstacle at ({obs[0]}, {obs[1]}): {min_dist:.3f} m")

    overall_min = min(min_distances)
    print(f"Overall minimum distance: {overall_min:.3f} m")
    print(f"Safety maintained: {'✅' if overall_min > d_min_obstacle else '❌'}")

    # QP solver performance (CBFKit provides this data)
    try:
        if "error" in c_keys:
            error_idx = c_keys.index("error")
            # c_values[error_idx] should be an array of booleans/ints
            solver_errors = int(np.sum(c_values[error_idx]))
            print(f"QP solver errors: {solver_errors}")
        else:
            print("No error data available")
    except Exception as e:
        print(f"Could not analyze solver errors: {e}")

    # Simulation statistics
    print(f"Simulation duration: {len(states) * 0.1:.1f} s")
    print(f"Total simulation steps: {len(states)}")

    # Velocity analysis
    velocities = states[:, 2]  # v is the 3rd component
    max_velocity = np.max(velocities)
    avg_velocity = np.mean(velocities)
    print(f"Max velocity: {max_velocity:.2f} m/s")
    print(f"Average velocity: {avg_velocity:.2f} m/s")


def main():
    """Main function demonstrating single robot CBF navigation."""

    # Run the simulation using CBFKit framework
    (
        states,
        controls,
        _,
        _,
        c_keys,
        c_values,
        desired_state,
        obstacles,
        d_min_obstacle,
    ) = run_simulation()

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_results(states, controls, desired_state, obstacles, d_min_obstacle)

    # Create animation
    create_animation(states, controls, desired_state, obstacles, d_min_obstacle)

    # Analyze performance
    analyze_performance(
        states, controls, c_keys, c_values, desired_state, obstacles, d_min_obstacle
    )

    print("\nResults saved to:")
    print("📊 Comprehensive plot: single_robot_cbf_comprehensive.png")
    print("📈 CBFKit data: single_robot_cbf_results.csv")
    print("🎬 Animation: single_robot_cbf_animation.mp4 (or .gif)")
    print("\n🎯 Successfully demonstrated single robot CBF navigation using CBFKit framework!")


if __name__ == "__main__":
    main()
