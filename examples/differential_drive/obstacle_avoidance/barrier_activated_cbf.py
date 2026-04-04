"""Differential drive obstacle avoidance with barrier-activated CBF.

Demonstrates a unicycle robot navigating among static obstacles using a
barrier-activated CBF-CLF QP controller. Only the k-closest barriers are
activated at each timestep, improving scalability for cluttered environments.
"""

import os

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle

from cbfkit.utils.animator import save_animation

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory

# Activated Controller Imports
from cbfkit.controllers.cbf_clf.barrier_activated_cbf_clf_qp_control_laws import (
    barrier_activated_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.controllers.cbf_clf.utils.barrier_activation import compute_activation_weights
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers

# Fixed imports to match project structure
from cbfkit.certificates import concatenate_certificates
from cbfkit.certificates import rectify_relative_degree
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor


def create_scenario():
    """Create CBF navigation scenario."""

    # Robot setup
    dynamics = unicycle.plant(l=1.0)
    dynamics.a_max = 5.0
    dynamics.omega_max = 5.0
    dynamics.v_max = 3.0
    dynamics.goal_tol = 0.2

    d_min = 0.7  # Safety distance
    init_state = jnp.array([0.5, 0.5, 0.0, 0.0])
    goal_state = jnp.array([11.0, 7.5, 0.0, 0.0])

    # Obstacles
    obstacles = [
        (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0),
        (4.0, 0.8, 0.0),
        (4.0, 2.8, 0.0),
        (8.0, 5.5, 0.0),
        (9.5, 6.5, 0.0),
        (3.0, 5.0, 0.0),
        (3.5, 6.5, 0.0),
        # (10.8, 5.5, 0.0),
    ]

    print(
        f"Robot setup: {len(obstacles)} obstacles, goal distance: {np.linalg.norm(goal_state[:2] - init_state[:2]):.1f}m"
    )

    # Controllers
    # Fix: Proportional controller returns a function, need to wrap it if passed as nominal_input
    # But cbf_controller handles it if it matches signature.
    # Check signature of proportional_controller in this repo.
    # It requires (t, state, key, desired_state)

    base_nom = unicycle.controllers.proportional_controller(
        dynamics=dynamics, Kp_pos=3.0, Kp_theta=4.0
    )

    from jax import random

    def nom_controller(t, state, key=None, x_des=None):
        # Proportional controller expects (t, state, key, desired_state)
        # If x_des is None (not provided by planner), use global goal_state
        target = x_des if x_des is not None else goal_state
        rng = key if key is not None else random.PRNGKey(0)
        return base_nom(t, state, rng, target)

    # Barriers
    # Use ellipsoidal_barrier_factory
    cbf_factory, _, _ = ellipsoidal_barrier_factory(
        system_position_indices=(0, 1),
        obstacle_position_indices=(0, 1),
        ellipsoid_axis_indices=(0, 1),
    )

    barriers = []
    for obs in obstacles:
        barrier = rectify_relative_degree(
            function=cbf_factory(jnp.array(obs), (d_min, d_min)),
            system_dynamics=dynamics,
            state_dim=4,
            form="exponential",
            roots=jnp.array([-1.0, -1.0]),
        )(
            certificate_conditions=zeroing_barriers.linear_class_k(10.0),
        )
        barriers.append(barrier)

    barrier_package = concatenate_certificates(*barriers)
    obstacle_positions = jnp.array([[obs[0], obs[1]] for obs in obstacles])

    # CBF controller with activation
    # We assume the controller signature matches cbf_clf_qp_generator's output
    # plus extra args like obstacle_positions
    controller = cbf_controller(
        control_limits=jnp.array([5.0, 5.0]),
        dynamics_func=dynamics,
        barriers=barrier_package,
        obstacle_positions=obstacle_positions,
        k_closest=4,
        activation_radius=3.0,
        relaxable_cbf=True,
    )

    return dynamics, controller, nom_controller, init_state, goal_state, obstacles, d_min


def run_simulation():
    """Run CBF navigation simulation."""

    print("CBF Navigation with Barrier Activation")

    (
        dynamics,
        controller,
        nom_controller,
        init_state,
        goal_state,
        obstacles,
        d_min,
    ) = create_scenario()

    print("Running simulation...")

    # sim.execute returns 8 values
    x, u, z, p, c_keys, c_values, p_keys, p_values = sim.execute(
        x0=init_state,
        dt=0.05,
        num_steps=500 if not os.getenv("CBFKIT_TEST_MODE") else 50,  # 25 seconds
        dynamics=dynamics,
        integrator=integrator,
        controller=controller,
        nominal_controller=nom_controller,
        sensor=sensor,
        estimator=estimator,
        filepath="examples/differential_drive/obstacle_avoidance/results/navigation_results",
        verbose=True,
    )

    return x, u, z, p, c_keys, c_values, goal_state, obstacles, d_min


def plot_results(states, controls, goal_state, obstacles, d_min):
    """Create visualization with trajectory phases."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Trajectory plot with phase coloring
    ax1.set_facecolor("#f8f9fa")

    # Plot trajectory with progress coloring
    n_segments = 4
    segment_length = len(states) // n_segments
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]  # Green to red progression

    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < n_segments - 1 else len(states)
        segment = states[start_idx:end_idx]

        if len(segment) > 0:
            ax1.plot(
                segment[:, 0],
                segment[:, 1],
                color=colors[i],
                linewidth=3,
                alpha=0.8,
                label=f"Phase {i+1}",
                zorder=3,
            )

    # Enhanced markers
    ax1.plot(
        states[0, 0],
        states[0, 1],
        "o",
        color="#27ae60",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=2,
        label="Start",
        zorder=5,
    )
    ax1.plot(
        goal_state[0],
        goal_state[1],
        "*",
        color="#f39c12",
        markersize=25,
        markeredgecolor="white",
        markeredgewidth=2,
        label="Goal",
        zorder=5,
    )

    # Enhanced obstacles with shadows
    for obs in obstacles:
        # Shadow effect
        shadow = Circle(
            (obs[0] + 0.05, obs[1] - 0.05), d_min, fill=True, facecolor="gray", alpha=0.2, zorder=1
        )
        ax1.add_patch(shadow)
        # Main obstacle
        circle = Circle(
            (obs[0], obs[1]),
            d_min,
            fill=True,
            facecolor="#e74c3c",
            alpha=0.7,
            edgecolor="darkred",
            linewidth=2,
            zorder=2,
        )
        ax1.add_patch(circle)
        ax1.plot(obs[0], obs[1], "s", color="darkred", markersize=6, zorder=4)

    ax1.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
    ax1.set_title("CBF Navigation with Barrier Activation", fontsize=14, fontweight="bold", pad=20)
    ax1.legend(loc="upper left", framealpha=0.9, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.4, linestyle="--")
    ax1.set_aspect("equal")

    # Enhanced velocity profile
    time = np.arange(len(states)) * 0.05
    ax2.set_facecolor("#f8f9fa")
    ax2.fill_between(time, 0, states[:, 2], alpha=0.3, color="#3498db", label="Velocity area")
    ax2.plot(time, states[:, 2], color="#2980b9", linewidth=2.5, label="Velocity")
    ax2.axhline(
        y=np.mean(states[:, 2]),
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Average: {np.mean(states[:, 2]):.2f} m/s",
    )
    ax2.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Velocity (m/s)", fontsize=12, fontweight="bold")
    ax2.set_title("Velocity Profile", fontsize=14, fontweight="bold", pad=20)
    ax2.legend(framealpha=0.9, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.4, linestyle="--")

    # Enhanced control inputs
    ax3.set_facecolor("#f8f9fa")
    if len(controls) > 0:
        time_ctrl = np.arange(len(controls)) * 0.05
        ax3.plot(
            time_ctrl,
            controls[:, 0],
            color="#e74c3c",
            linewidth=2.5,
            label="Acceleration",
            alpha=0.8,
        )
        ax3.plot(
            time_ctrl,
            controls[:, 1],
            color="#27ae60",
            linewidth=2.5,
            label="Angular velocity",
            alpha=0.8,
        )

        # Add control bounds visualization
        ax3.axhline(y=100, color="red", linestyle=":", alpha=0.5, label="Control limits")
        ax3.axhline(y=-100, color="red", linestyle=":", alpha=0.5)

        ax3.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Control Input", fontsize=12, fontweight="bold")
        ax3.set_title("Control Inputs", fontsize=14, fontweight="bold", pad=20)
        ax3.legend(framealpha=0.9, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.4, linestyle="--")

    # Enhanced safety analysis
    ax4.set_facecolor("#f8f9fa")
    min_distances = []
    for state in states:
        distances = [np.linalg.norm(state[:2] - np.array(obs[:2])) for obs in obstacles]
        min_distances.append(min(distances))

    # Color-coded safety zones
    ax4.fill_between(time, 0, d_min, alpha=0.3, color="red", label="Danger zone")
    ax4.fill_between(time, d_min, d_min * 1.5, alpha=0.2, color="orange", label="Caution zone")

    ax4.plot(
        time, min_distances, color="#2c3e50", linewidth=3, label="Min obstacle distance", zorder=3
    )
    ax4.axhline(
        y=d_min,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Safety threshold ({d_min}m)",
        alpha=0.8,
    )

    # Mark violations
    violations = [i for i, d in enumerate(min_distances) if d < d_min]
    if violations:
        violation_times = [time[i] for i in violations]
        violation_dists = [min_distances[i] for i in violations]
        ax4.scatter(
            violation_times,
            violation_dists,
            color="red",
            s=50,
            zorder=4,
            label=f"Violations ({len(violations)})",
        )

    ax4.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Distance (m)", fontsize=12, fontweight="bold")
    ax4.set_title("Safety Monitoring", fontsize=14, fontweight="bold", pad=20)
    ax4.legend(framealpha=0.9, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.4, linestyle="--")

    plt.tight_layout(pad=3.0)
    filename = "examples/differential_drive/obstacle_avoidance/results/navigation_analysis.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Enhanced analysis saved to {filename}")


def create_animation(states, goal_state, obstacles, d_min):
    """Create enhanced animation with barrier activation visualization."""

    print("Creating animation...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#f8f9fa")

    # Setup bounds with better margins
    all_x = [state[0] for state in states] + [obs[0] for obs in obstacles] + [goal_state[0]]
    all_y = [state[1] for state in states] + [obs[1] for obs in obstacles] + [goal_state[1]]
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_xlabel("X Position (m)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y Position (m)", fontsize=14, fontweight="bold")
    ax.set_title(
        "CBF Navigation with Dynamic Barrier Activation", fontsize=16, fontweight="bold", pad=20
    )
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.set_aspect("equal")

    obstacle_positions = jnp.array([[obs[0], obs[1]] for obs in obstacles])

    # Create enhanced obstacle circles with shadows
    obstacle_circles = []
    obstacle_shadows = []
    obstacle_centers = []

    for obs in obstacles:
        # Shadow effect
        shadow = Circle(
            (obs[0] + 0.05, obs[1] - 0.05), d_min, fill=True, facecolor="gray", alpha=0.1, zorder=1
        )
        ax.add_patch(shadow)
        obstacle_shadows.append(shadow)

        # Main obstacle circle
        circle = Circle(
            (obs[0], obs[1]),
            d_min,
            fill=True,
            facecolor="red",
            alpha=0.3,
            edgecolor="darkred",
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(circle)
        obstacle_circles.append(circle)

        # Center marker
        center = ax.plot(obs[0], obs[1], "s", color="darkred", markersize=6, zorder=4)[0]
        obstacle_centers.append(center)

    # Enhanced goal visualization
    goal_outer = Circle(
        (goal_state[0], goal_state[1]),
        0.4,
        fill=False,
        edgecolor="#f39c12",
        linewidth=3,
        alpha=0.8,
        zorder=1,
    )
    ax.add_patch(goal_outer)
    goal_circle = Circle(
        (goal_state[0], goal_state[1]), 0.25, fill=True, facecolor="#f39c12", alpha=0.9, zorder=2
    )
    ax.add_patch(goal_circle)
    ax.plot(goal_state[0], goal_state[1], "*", color="white", markersize=20, zorder=3)

    # Start position marker
    start_circle = Circle(
        (states[0][0], states[0][1]),
        0.2,
        fill=True,
        facecolor="#27ae60",
        alpha=0.8,
        edgecolor="white",
        linewidth=2,
        zorder=2,
    )
    ax.add_patch(start_circle)

    # Enhanced robot visualization
    robot_circle = Circle(
        (0, 0),
        0.25,
        fill=True,
        facecolor="#3498db",
        alpha=0.9,
        edgecolor="white",
        linewidth=2,
        zorder=5,
    )
    ax.add_patch(robot_circle)

    # Robot direction arrow
    arrow_ref = [None]

    # Enhanced trail with gradient effect
    (trail_line,) = ax.plot([], [], color="#2980b9", alpha=0.7, linewidth=3, zorder=3)
    (trail_fade,) = ax.plot([], [], color="#85c1e9", alpha=0.4, linewidth=1.5, zorder=2)

    # Enhanced text displays
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="#ecf0f1",
            alpha=0.9,
            edgecolor="#34495e",
            linewidth=1,
        ),
    )

    status_text = ax.text(
        0.02,
        0.85,
        "",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#e8f6f3",
            alpha=0.9,
            edgecolor="#16a085",
            linewidth=1,
        ),
    )

    activation_text = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#fdf2e9",
            alpha=0.9,
            edgecolor="#e67e22",
            linewidth=1,
        ),
    )

    # Enhanced legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.8, label="Active Barriers"),
        Patch(facecolor="gray", alpha=0.3, label="Inactive Barriers"),
        Patch(facecolor="#3498db", alpha=0.9, label="Robot"),
        Patch(facecolor="#f39c12", alpha=0.9, label="Goal"),
        Patch(facecolor="#2980b9", alpha=0.7, label="Trajectory"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        framealpha=0.9,
        fancybox=True,
        shadow=True,
        fontsize=11,
    )

    def animate_frame(frame):
        if frame >= len(states):
            return (
                [robot_circle, trail_line, trail_fade, time_text, status_text, activation_text]
                + obstacle_circles
                + obstacle_centers
                + [arrow_ref[0]]
                if arrow_ref[0]
                else []
            )

        state = states[frame]
        x, y, v, theta = state[0], state[1], state[2], state[3]
        t = frame * 0.05

        # Update displays
        time_text.set_text(f"Time: {t:.1f}s\nFrame: {frame}/{len(states)}")
        robot_circle.center = (x, y)

        # Compute activation weights
        activation_weights = compute_activation_weights(
            state, obstacle_positions, activation_type="combined", k=3, radius=2.0, smoothness=5.0
        )

        # Update obstacle appearance based on activation
        active_count = 0
        max_weight = 0
        for circle, center, weight in zip(obstacle_circles, obstacle_centers, activation_weights):
            weight_val = float(weight)
            max_weight = max(max_weight, weight_val)

            if weight_val > 0.01:  # Active threshold
                # Active obstacles: vibrant red with intensity based on weight
                intensity = 0.6 + 0.4 * weight_val
                circle.set_facecolor("#e74c3c")
                circle.set_alpha(intensity)
                circle.set_edgecolor("#c0392b")
                circle.set_linewidth(3)
                center.set_color("#a93226")
                center.set_markersize(8)
                active_count += 1
            else:
                # Inactive obstacles: muted gray
                circle.set_facecolor("#95a5a6")
                circle.set_alpha(0.25)
                circle.set_edgecolor("#7f8c8d")
                circle.set_linewidth(1)
                center.set_color("#7f8c8d")
                center.set_markersize(4)

        # Enhanced robot direction arrow
        arrow_length = 0.4
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)

        if arrow_ref[0] is not None:
            arrow_ref[0].remove()
        arrow_ref[0] = Arrow(x, y, dx, dy, width=0.25, color="white", alpha=0.9, zorder=6)
        ax.add_patch(arrow_ref[0])

        # Enhanced trail with fade effect
        trail_length = min(frame + 1, 100)  # Show last 100 points
        start_idx = max(0, frame + 1 - trail_length)

        trail_x = [s[0] for s in states[start_idx : frame + 1]]
        trail_y = [s[1] for s in states[start_idx : frame + 1]]
        trail_line.set_data(trail_x, trail_y)

        # Fade trail (older part)
        if frame > 50:
            fade_x = [s[0] for s in states[max(0, frame - 100) : frame - 50]]
            fade_y = [s[1] for s in states[max(0, frame - 100) : frame - 50]]
            trail_fade.set_data(fade_x, fade_y)

        # Enhanced status information
        goal_dist = np.linalg.norm(state[:2] - goal_state[:2])
        min_obs_dist = min(np.linalg.norm(state[:2] - np.array(obs[:2])) for obs in obstacles)

        status_text.set_text(
            f"Goal: {goal_dist:.2f}m\n"
            f"Speed: {v:.2f}m/s\n"
            f"Clearance: {min_obs_dist:.2f}m\n"
            f"Heading: {np.degrees(theta):.0f}°"
        )

        # Enhanced activation information
        activation_text.set_text(
            f"Barrier Activation\n"
            f"Active: {active_count}/{len(obstacles)} barriers\n"
            f"Max weight: {max_weight:.3f}\n"
            f"Mode: k=3 + smooth(r=2.0)\n"
            f'Safety: {"OK" if min_obs_dist >= d_min else "WARNING"}'
        )

        return (
            [
                robot_circle,
                arrow_ref[0],
                trail_line,
                trail_fade,
                time_text,
                status_text,
                activation_text,
            ]
            + obstacle_circles
            + obstacle_centers
        )

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=len(states), interval=50, blit=False, repeat=True
    )

    # Save animation
    save_animation(
        anim, "examples/differential_drive/obstacle_avoidance/results/navigation_animation.mp4"
    )

    plt.close()
    return anim


def analyze_performance(states, _, goal_state, obstacles, d_min):
    """Analyze navigation performance."""

    print("\nPerformance Analysis")

    final_state = states[-1]
    goal_error = np.linalg.norm(final_state[:2] - goal_state[:2])

    # Safety check
    min_distances = []
    for state in states:
        distances = [np.linalg.norm(state[:2] - np.array(obs[:2])) for obs in obstacles]
        min_distances.append(min(distances))

    overall_min = min(min_distances)
    safety_violations = sum(1 for d in min_distances if d < d_min)

    # Performance rating
    if goal_error < 0.5 and overall_min >= d_min:
        performance = "EXCELLENT"
    elif goal_error < 1.0 and overall_min >= d_min * 0.9:
        performance = "GOOD"
    else:
        performance = "NEEDS IMPROVEMENT"

    print(f"Goal error: {goal_error:.2f}m")
    print(f"Min clearance: {overall_min:.2f}m")
    print(f"Safety violations: {safety_violations}")
    print(f"Performance: {performance}")
    print(f"Time: {len(states) * 0.05:.1f}s")

    return {"goal_error": goal_error, "min_clearance": overall_min, "performance": performance}


def main():
    """Main function."""

    # Run simulation
    states, controls, _, _, _, _, goal_state, obstacles, d_min = run_simulation()

    # Generate enhanced outputs
    if not os.getenv("CBFKIT_TEST_MODE"):
        print("\nCreating enhanced visualizations...")
        plot_results(states, controls, goal_state, obstacles, d_min)
        create_animation(states, goal_state, obstacles, d_min)

    # Analyze results
    performance = analyze_performance(states, controls, goal_state, obstacles, d_min)

    print("\nResults saved:")
    print("Enhanced analysis: navigation_analysis.png")
    print("HD animation: navigation_animation.mp4/.gif")
    print("Simulation data: navigation_results.csv")
    print(f"\nNavigation complete! {performance['performance']}")


if __name__ == "__main__":
    main()
