"""
Dynamic Obstacle Avoidance with CBF for Differential Drive Robot.

This example demonstrates a unicycle robot navigating to a goal while avoiding
moving (dynamic) obstacles using Control Barrier Functions (CBF).
The barriers are formulated on the augmented state-time space to rigorously
account for the time-varying nature of the constraints.
"""

import os
import sys
import time

# Add project root to path
sys.path.append(os.getcwd())

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf.vanilla_cbf_clf_qp_control_laws import (
    vanilla_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor

# --- Dynamic Obstacle Definitions ---


def get_obstacle_pos(t, obs_idx):
    """Returns the position of the dynamic obstacle at time t."""
    if obs_idx == 0:
        # Obstacle 0: Horizontal oscillation
        # Center (5, 2), amplitude 2.0, period 10s
        x = 5.0 + 2.0 * jnp.sin(2 * jnp.pi * t / 10.0)
        y = 2.0
        return jnp.array([x, y])
    elif obs_idx == 1:
        # Obstacle 1: Vertical oscillation
        # Center (5, 5), amplitude 2.0, period 8s
        x = 5.0
        y = 5.0 + 2.0 * jnp.cos(2 * jnp.pi * t / 8.0)
        return jnp.array([x, y])
    elif obs_idx == 2:
        # Obstacle 2: Circular motion
        # Center (8, 4), radius 1.5, period 12s
        omega = 2 * jnp.pi / 12.0
        x = 8.0 + 1.5 * jnp.cos(omega * t)
        y = 4.0 + 1.5 * jnp.sin(omega * t)
        return jnp.array([x, y])
    else:
        return jnp.array([-100.0, -100.0])


def create_dynamic_barrier_func(obs_idx, d_min):
    """Creates a barrier function h(z) where z = [x, y, v, theta, t]."""

    def h(z):
        # Extract state and time
        pos = z[:2]
        t = z[-1]

        # Get obstacle position
        obs_pos = get_obstacle_pos(t, obs_idx)

        # Compute distance squared minus safety squared
        # h(x, t) = ||x - x_obs(t)||^2 - d_min^2
        diff = pos - obs_pos
        return jnp.dot(diff, diff) - d_min**2

    return h


def run_simulation():
    """Sets up and runs the simulation."""
    print("Setting up Dynamic Obstacle Avoidance Scenario...")

    # 1. Dynamics
    dynamics = unicycle.plant(l=1.0)
    dynamics.a_max = 5.0
    dynamics.omega_max = 5.0
    dynamics.v_max = 4.0

    # 2. Scenario
    init_state = jnp.array([0.0, 0.0, 0.0, 0.0])
    goal_state = jnp.array([10.0, 7.0, 0.0, 0.0])
    d_min = 1.0  # Radius of obstacle + robot radius + margin
    num_obstacles = 3

    # 3. Nominal Controller
    # Proportional controller to goal
    from cbfkit.systems.unicycle.models.accel_unicycle.controllers.proportional_controller import (
        proportional_controller,
    )

    # Create the base proportional controller
    base_nom = proportional_controller(dynamics, Kp_pos=2.0, Kp_theta=2.0)

    # Wrap it to handle the signature required by sim.execute/controller
    # The vanilla controller passes u_nom computed by nominal_controller
    # sim.execute calls nominal_controller(t, x)
    def nom_controller(t, x, key=None, data=None):
        return base_nom(t, x, key, goal_state)

    # 4. Barriers
    print("Rectifying relative degrees for dynamic barriers...")
    barriers = []
    for i in range(num_obstacles):
        # Create h(z)
        h_z = create_dynamic_barrier_func(i, d_min)

        # Rectify (relative degree 2 for acceleration unicycle)
        # We pass nominal dynamics and state_dim=4 (physical state dim)
        # cbfkit handles time augmentation internally
        barrier_rectified = rectify_relative_degree(
            function=h_z,
            system_dynamics=dynamics,
            state_dim=4,
            form="exponential",
        )(
            certificate_conditions=zeroing_barriers.linear_class_k(2.0),
        )

        barriers.append(barrier_rectified)

    barrier_package = concatenate_certificates(*barriers)

    # 5. CBF Controller
    print("Initializing CBF Controller...")
    controller = cbf_controller(
        control_limits=jnp.array([dynamics.a_max, dynamics.omega_max]),
        dynamics_func=dynamics,
        barriers=barrier_package,
        # We don't use 'obstacle_positions' arg here as barriers handle it internally
    )

    # 6. Simulation
    if os.getenv("CBFKIT_TEST_MODE"):
        dt = 0.05
        tf = 1.0
    else:
        dt = 0.05
        tf = 20.0
    num_steps = int(tf / dt)

    print(f"Starting simulation ({tf}s)...")
    start_time = time.time()

    x, u, z, p, c_keys, c_values, p_keys, p_values = sim.execute(
        x0=init_state,
        dt=dt,
        num_steps=num_steps,
        dynamics=dynamics,
        integrator=integrator,
        controller=controller,
        nominal_controller=nom_controller,
        sensor=sensor,
        estimator=estimator,
        filepath="examples/differential_drive/results/dynamic_obstacle_results",
        verbose=True,
    )

    print(f"Simulation finished in {time.time() - start_time:.2f}s")
    return x, u, goal_state, d_min, num_obstacles, dt


def create_visualization(states, controls, goal_state, d_min, num_obstacles, dt):
    """Creates animation and plots."""
    print("Generating visualization...")

    # Convert JAX arrays to NumPy
    states = np.array(states)
    controls = np.array(controls)
    goal_state = np.array(goal_state)

    # --- Plot 1: Static Trajectory with Obstacle Trails ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot Robot Trajectory
    ax.plot(states[:, 0], states[:, 1], "b-", linewidth=2, label="Robot Path")
    ax.plot(states[0, 0], states[0, 1], "go", label="Start")
    ax.plot(goal_state[0], goal_state[1], "r*", markersize=15, label="Goal")

    # Plot Obstacle Paths
    time_vec = np.arange(len(states)) * dt
    for i in range(num_obstacles):
        obs_pos = np.array([get_obstacle_pos(t, i) for t in time_vec])
        ax.plot(obs_pos[:, 0], obs_pos[:, 1], "--", alpha=0.5, label=f"Obstacle {i} Path")

    ax.set_title("Robot Trajectory and Dynamic Obstacle Paths")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")

    plt.savefig("examples/differential_drive/results/dynamic_obstacle_trajectory.png")
    plt.close()

    # --- Animation ---
    print("Creating animation...")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#f8f9fa")

    # Bounds
    all_x = list(states[:, 0]) + [goal_state[0]]
    all_y = list(states[:, 1]) + [goal_state[1]]
    margin = 2.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Dynamic Obstacle Avoidance", fontsize=16)

    # Static Elements
    # Goal
    goal_circle = Circle((goal_state[0], goal_state[1]), 0.3, color="gold", alpha=0.5, zorder=1)
    ax.add_patch(goal_circle)
    ax.plot(goal_state[0], goal_state[1], "*", color="orange", markersize=15)

    # Dynamic Elements
    robot_circle = Circle((0, 0), 0.3, color="blue", zorder=5)
    ax.add_patch(robot_circle)

    obs_circles = []
    for i in range(num_obstacles):
        c = Circle((0, 0), d_min, color="red", alpha=0.3, zorder=4)
        ax.add_patch(c)
        obs_circles.append(c)

    (robot_trail,) = ax.plot([], [], "b-", alpha=0.5, linewidth=1)

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def animate(frame):
        t = frame * dt
        state = states[frame]

        # Update Robot
        robot_circle.center = (state[0], state[1])

        # Update Trail
        robot_trail.set_data(states[:frame, 0], states[:frame, 1])

        # Update Obstacles
        for i, c in enumerate(obs_circles):
            pos = get_obstacle_pos(t, i)
            c.center = (pos[0], pos[1])

        # Update Text
        dist_to_goal = np.linalg.norm(state[:2] - goal_state[:2])
        time_text.set_text(f"Time: {t:.1f}s\nGoal Dist: {dist_to_goal:.1f}m")

        return [robot_circle, robot_trail, time_text] + obs_circles

    anim = animation.FuncAnimation(fig, animate, frames=len(states), interval=50, blit=True)

    try:
        anim.save(
            "examples/differential_drive/results/dynamic_obstacle_animation.mp4",
            writer="ffmpeg",
            fps=20,
        )
        print("Saved animation to dynamic_obstacle_animation.mp4")
    except Exception as e:
        print(f"MP4 save failed, trying GIF: {e}")
        anim.save(
            "examples/differential_drive/results/dynamic_obstacle_animation.gif",
            writer="pillow",
            fps=15,
        )
        print("Saved animation to dynamic_obstacle_animation.gif")

    plt.close()


def main():
    # Ensure results directory exists
    os.makedirs("examples/differential_drive/results", exist_ok=True)

    x, u, goal_state, d_min, num_obstacles, dt = run_simulation()
    if not os.getenv("CBFKIT_TEST_MODE"):
        create_visualization(x, u, goal_state, d_min, num_obstacles, dt)

    final_dist = np.linalg.norm(x[-1, :2] - goal_state[:2])
    print(f"\nFinal distance to goal: {final_dist:.2f}m")
    if final_dist < 0.5:
        print("SUCCESS: Goal reached!")
    else:
        print("Note: Goal not reached (check simulation time or constraints)")


if __name__ == "__main__":
    main()
