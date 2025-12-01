"""
Augmented State Dynamic Obstacle Avoidance.

This example demonstrates avoiding dynamic obstacles by augmenting the system state.
Instead of treating obstacles as time-varying parameters, we model them as part of
the system dynamics (autonomous agents). This allows defining barrier functions
directly on the joint state space (robot + obstacles), enabling standard CBF
machinery (like rectify_relative_degree) to automatically account for obstacle motion.
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
from cbfkit.controllers.cbf_clf.utils.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf.utils.certificate_packager import concatenate_certificates
from cbfkit.controllers.cbf_clf.utils.rectify_relative_degree import rectify_relative_degree
from cbfkit.controllers.cbf_clf.vanilla_cbf_clf_qp_control_laws import (
    vanilla_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor

# --- Augmented Dynamics Helpers ---


def create_augmented_dynamics(robot_dynamics, obstacle_dynamics_list):
    """
    Combines robot dynamics and a list of obstacle dynamics into a single system.

    Args:
        robot_dynamics: Callable returning (f, g) for the robot (dim 4).
        obstacle_dynamics_list: List of callables, each returning f_obs (dim 2) for an obstacle.
                                Obstacles are assumed to be autonomous (g=0).

    Returns:
        augmented_dynamics: Callable returning (F, G) for the combined state.
    """

    def dynamics(z):
        # 1. Robot Dynamics (Indices 0-3)
        x_robot = z[:4]
        f_r, g_r = robot_dynamics(x_robot)

        f_list = [f_r]
        g_rows = [g_r]

        # 2. Obstacle Dynamics
        current_idx = 4
        for obs_dyn in obstacle_dynamics_list:
            # Assuming each obstacle is 2D (x, y)
            # If obstacles have more states, this needs adjustment
            x_obs = z[current_idx : current_idx + 2]
            f_o = obs_dyn(x_obs)

            f_list.append(f_o)
            # Obstacles are not controlled by the robot's input
            g_rows.append(jnp.zeros((2, g_r.shape[1])))

            current_idx += 2

        # Concatenate
        F = jnp.concatenate(f_list)
        G = jnp.concatenate(g_rows, axis=0)

        return F, G

    return dynamics


# --- Obstacle Dynamics Models ---


def circular_obstacle_dynamics(center, omega):
    """Returns dynamics for an obstacle moving in a circle."""
    c_x, c_y = center

    def f(x):
        # x = [px, py]
        px, py = x[0], x[1]
        # velocity vector tangent to circle
        vx = -omega * (py - c_y)
        vy = omega * (px - c_x)
        return jnp.array([vx, vy])

    return f


def linear_obstacle_dynamics(velocity):
    """Returns dynamics for an obstacle moving with constant velocity."""
    v = jnp.array(velocity)

    def f(x):
        return v

    return f


# --- Barrier Functions ---


def create_augmented_barrier(robot_pos_idx, obs_pos_idx, d_min):
    """
    Creates a barrier function h(z) = ||p_r - p_o||^2 - d_min^2.
    """

    def h(z):
        p_r = z[robot_pos_idx : robot_pos_idx + 2]
        p_o = z[obs_pos_idx : obs_pos_idx + 2]
        diff = p_r - p_o
        return jnp.dot(diff, diff) - d_min**2

    return h


def run_simulation():
    print("Setting up Augmented State Simulation...")

    # 1. Define Systems
    # Robot
    robot_dyn = unicycle.plant(l=1.0)
    robot_dyn.a_max = 5.0
    robot_dyn.omega_max = 5.0
    robot_dyn.v_max = 4.0

    # Obstacles
    # Obs 1: Circular path around (5, 4)
    obs1_dyn = circular_obstacle_dynamics(center=(5.0, 4.0), omega=0.5)
    obs1_init = jnp.array([5.0 + 2.0, 4.0])  # Start at radius 2.0

    # Obs 2: Crossing linearly
    obs2_dyn = linear_obstacle_dynamics(velocity=(-0.5, 0.2))
    obs2_init = jnp.array([9.0, 2.0])

    # Create Augmented System
    # Order in state z: [Robot(4), Obs1(2), Obs2(2)]
    # Total dim = 8
    aug_dynamics = create_augmented_dynamics(robot_dyn, [obs1_dyn, obs2_dyn])

    # Initial Augmented State
    # Robot starts at (0,0)
    robot_init = jnp.array([0.0, 0.0, 0.0, 0.0])
    z0 = jnp.concatenate([robot_init, obs1_init, obs2_init])

    goal_state = jnp.array([10.0, 8.0, 0.0, 0.0])
    d_min = 1.5

    # 2. Nominal Controller
    # The nominal controller needs to extract the robot state from the augmented state
    from cbfkit.systems.unicycle.models.accel_unicycle.controllers.proportional_controller import (
        proportional_controller,
    )

    base_nom = proportional_controller(robot_dyn, Kp_pos=2.0, Kp_theta=2.0)

    def nom_controller(t, z, key=None, data=None):
        # Extract robot state (first 4 elements)
        x_robot = z[:4]
        # Compute control
        u_nom, _ = base_nom(t, x_robot, key, goal_state)
        return u_nom, {}

    # 3. Barriers
    print("Rectifying barriers on augmented state...")
    barriers = []

    # Barrier for Obs 1
    # Robot pos indices: 0, 1
    # Obs 1 pos indices: 4, 5
    h1 = create_augmented_barrier(0, 4, d_min)
    b1 = rectify_relative_degree(
        function=h1,
        system_dynamics=aug_dynamics,
        state_dim=8,  # Full augmented dimension
        form="exponential",
    )(
        certificate_conditions=zeroing_barriers.linear_class_k(1.0),
        obstacle=jnp.array([]),  # Unused
        ellipsoid=(1.0, 1.0),  # Unused
    )
    barriers.append(b1)

    # Barrier for Obs 2
    # Robot pos indices: 0, 1
    # Obs 2 pos indices: 6, 7
    h2 = create_augmented_barrier(0, 6, d_min)
    b2 = rectify_relative_degree(
        function=h2, system_dynamics=aug_dynamics, state_dim=8, form="exponential"
    )(
        certificate_conditions=zeroing_barriers.linear_class_k(1.0),
        obstacle=jnp.array([]),
        ellipsoid=(1.0, 1.0),
    )
    barriers.append(b2)

    barrier_package = concatenate_certificates(*barriers)

    # 4. CBF Controller
    print("Initializing Controller...")
    controller = cbf_controller(
        control_limits=jnp.array([robot_dyn.a_max, robot_dyn.omega_max]),
        dynamics_func=aug_dynamics,  # Controller uses augmented dynamics!
        barriers=barrier_package,
    )

    # 5. Simulation
    dt = 0.05
    tf = 15.0
    num_steps = int(tf / dt)

    print("Starting Simulation...")
    start_time = time.time()

    # We pass aug_dynamics to simulator
    # Simulator will integrate the full state z (robot + obstacles)
    x, u, z_sim, p, c_keys, c_values, p_keys, p_values = sim.execute(
        x0=z0,
        dt=dt,
        num_steps=num_steps,
        dynamics=aug_dynamics,
        integrator=integrator,
        controller=controller,
        nominal_controller=nom_controller,
        sensor=sensor,
        estimator=estimator,
        filepath="examples/differential_drive/results/augmented_results",
        verbose=True,
    )

    print(f"Done in {time.time() - start_time:.2f}s")
    return x, u, goal_state, d_min


def create_visualization(states, goal_state, d_min):
    print("Creating Augmented System Animation...")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Extract trajectories
    # Robot: indices 0, 1
    rx = states[:, 0]
    ry = states[:, 1]

    # Obs 1: indices 4, 5
    o1x = states[:, 4]
    o1y = states[:, 5]

    # Obs 2: indices 6, 7
    o2x = states[:, 6]
    o2y = states[:, 7]

    # Set limits
    all_x = np.concatenate([rx, o1x, o2x, [goal_state[0]]])
    all_y = np.concatenate([ry, o1y, o2y, [goal_state[1]]])
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Elements
    robot_patch = Circle((0, 0), 0.3, color="blue", label="Robot")
    goal_patch = Circle((goal_state[0], goal_state[1]), 0.3, color="green", alpha=0.5, label="Goal")

    obs1_patch = Circle((0, 0), d_min, color="red", alpha=0.3, label="Dynamic Obs 1")
    obs2_patch = Circle((0, 0), d_min, color="orange", alpha=0.3, label="Dynamic Obs 2")

    ax.add_patch(robot_patch)
    ax.add_patch(goal_patch)
    ax.add_patch(obs1_patch)
    ax.add_patch(obs2_patch)

    (robot_trail,) = ax.plot([], [], "b-", alpha=0.5)

    ax.legend()
    ax.set_title("Augmented State Dynamic Avoidance")

    def animate(i):
        # Update Robot
        robot_patch.center = (rx[i], ry[i])
        robot_trail.set_data(rx[:i], ry[:i])

        # Update Obstacles
        obs1_patch.center = (o1x[i], o1y[i])
        obs2_patch.center = (o2x[i], o2y[i])

        return [robot_patch, robot_trail, obs1_patch, obs2_patch]

    anim = animation.FuncAnimation(fig, animate, frames=len(states), interval=50, blit=True)

    save_path = "examples/differential_drive/results/augmented_dynamic_animation.mp4"
    try:
        anim.save(save_path, writer="ffmpeg", fps=20)
        print(f"Saved to {save_path}")
    except Exception:
        anim.save(save_path.replace("mp4", "gif"), writer="pillow", fps=20)
        print(f"Saved to {save_path.replace('mp4', 'gif')}")

    plt.close()


def main():
    os.makedirs("examples/differential_drive/results", exist_ok=True)
    x, u, goal, d_min = run_simulation()
    create_visualization(x, goal, d_min)

    final_dist = np.linalg.norm(x[-1, :2] - goal[:2])
    print(f"Final Goal Distance: {final_dist:.3f}m")


if __name__ == "__main__":
    main()
