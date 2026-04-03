"""
Human-Aware Navigation with MPPI and CBF Safety Filter.

This script demonstrates a robot navigating among moving humans.
- **System**: Augmented state (Robot + N Humans).
- **Humans**: Modeled as Constant Velocity (CV) agents. State: [x, y, vx, vy].
- **Planner (Nominal)**: MPPI. Predicts human motion using CV model and plans robot trajectory.
- **Safety (Filter)**: CBF-QP. Ensures safety relative to the *augmented* dynamics (accounting for human velocity).
"""

import os
import sys
import time

# Add project root to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from jax import jit
from matplotlib.patches import Arrow, Circle

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf.vanilla_cbf_clf_qp_control_laws import (
    vanilla_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.controllers.mppi.mppi_generator import mppi_generator
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.utils.user_types import PlannerData

# --- 1. Dynamics Definitions ---


def human_cv_dynamics():
    """
    Constant Velocity Human Dynamics.
    State: [x, y, vx, vy]
    f = [vx, vy, 0, 0]
    g = 0
    """

    def f(x):
        # x: [px, py, vx, vy]
        return jnp.array([x[2], x[3], 0.0, 0.0])

    return f


def create_augmented_dynamics(robot_dyn, num_humans):
    """
    Creates augmented dynamics for Robot + N Humans.
    State Z: [Robot(4), Human1(4), Human2(4), ...]
    """
    human_f = human_cv_dynamics()

    def dynamics(z):
        # Robot
        x_r = z[:4]
        f_r, g_r = robot_dyn(x_r)

        f_list = [f_r]
        g_rows = [g_r]

        # Humans
        for i in range(num_humans):
            idx = 4 + i * 4
            x_h = z[idx : idx + 4]
            f_h = human_f(x_h)
            f_list.append(f_h)
            g_rows.append(jnp.zeros((4, g_r.shape[1])))

        F = jnp.concatenate(f_list)
        G = jnp.concatenate(g_rows, axis=0)

        return F, G

    return dynamics


# --- 2. MPPI Costs ---


def create_mppi_costs(goal_state, num_humans, d_safe):
    """
    Creates stage and terminal costs for MPPI.
    """
    R_val = 0.01
    Q_goal = 2.0
    Q_obs = 100.0
    Q_vel = 50.0  # Strong penalty for negative velocity (going backward)
    Q_heading = 5.0  # Penalty for heading misalignment with goal
    Q_accel = 20.0  # Penalty for negative acceleration when velocity is low

    @jit
    def stage_cost(z, u):
        # 1. Control Cost
        c_u = R_val * jnp.dot(u, u)

        # 2. Goal Cost (Robot only)
        p_r = z[:2]
        p_g = goal_state[:2]
        dist_goal = jnp.linalg.norm(p_r - p_g)
        c_goal = Q_goal * dist_goal

        # 3. Obstacle Cost (Humans)
        c_obs = 0.0
        for i in range(num_humans):
            idx = 4 + i * 4
            p_h = z[idx : idx + 2]
            dist_h = jnp.linalg.norm(p_r - p_h)
            # Soft constraint / Penalty
            # If dist < d_safe * 1.5, penalize
            c_obs += Q_obs * jnp.exp(-2.0 * (dist_h - d_safe))

        # 4. Velocity Direction Cost - penalize negative velocity (going backward)
        # Robot state: [x, y, v, theta] at indices 0:4
        v_robot = z[2]
        # Penalize negative velocity strongly to prevent backward motion
        c_vel = Q_vel * jnp.maximum(0.0, -v_robot)

        # 5. Heading Alignment Cost - penalize heading that doesn't point toward goal
        theta_robot = z[3]
        # Compute desired heading to goal
        delta = p_g - p_r
        theta_goal = jnp.arctan2(delta[1], delta[0])
        # Heading error (wrapped to [-pi, pi])
        heading_error = theta_robot - theta_goal
        heading_error = jnp.arctan2(jnp.sin(heading_error), jnp.cos(heading_error))
        c_heading = Q_heading * heading_error**2

        # 6. Forward Acceleration Bias - penalize negative acceleration when velocity is low
        # This encourages the robot to accelerate forward initially
        a_robot = u[0]  # Acceleration control
        # Penalize negative acceleration more when velocity is near zero or negative
        vel_factor = jnp.exp(-2.0 * v_robot)  # Higher weight when v is small/negative
        c_accel = Q_accel * vel_factor * jnp.maximum(0.0, -a_robot)

        return c_u + c_goal + c_obs + c_vel + c_heading + c_accel

    @jit
    def terminal_cost(z):
        p_r = z[:2]
        p_g = goal_state[:2]
        return 10.0 * jnp.linalg.norm(p_r - p_g) ** 2

    return stage_cost, terminal_cost


# --- 3. Barrier Functions ---


def create_human_barrier(human_idx, d_min):
    """
    h(z) = ||p_r - p_h||^2 - d_min^2
    """
    idx = 4 + human_idx * 4

    def h(z):
        p_r = z[:2]
        p_h = z[idx : idx + 2]
        diff = p_r - p_h
        return jnp.dot(diff, diff) - d_min**2

    return h


# --- 4. Simulation Setup ---


def run_simulation():
    print("Setting up Human-Aware Navigation (MPPI + CBF)...")

    # Parameters
    d_safe = 1.0
    num_humans = 2
    dt = 0.1
    tf = 10.0 if not os.getenv("CBFKIT_TEST_MODE") else 1.0

    # Robot Dynamics
    robot_dyn = unicycle.plant(l=1.0)
    robot_dyn.a_max = 5.0
    robot_dyn.omega_max = 5.0
    robot_dyn.v_max = 4.0

    # Augmented Dynamics
    aug_dynamics = create_augmented_dynamics(robot_dyn, num_humans)

    # Initial States
    # Robot: (0, 4), facing right
    x0_r = jnp.array([0.0, 4.0, 0.0, 0.0])

    # Human 1: (5, 0), moving Up (vx=0, vy=1.0)
    x0_h1 = jnp.array([5.0, 0.0, 0.0, 1.0])

    # Human 2: (8, 8), moving Left-Down (vx=-0.5, vy=-0.8)
    x0_h2 = jnp.array([8.0, 8.0, -0.5, -0.8])

    z0 = jnp.concatenate([x0_r, x0_h1, x0_h2])
    goal_state = jnp.array([10.0, 4.0, 0.0, 0.0])

    # --- MPPI Planner ---
    print("Initializing MPPI Planner...")
    stage_cost, terminal_cost = create_mppi_costs(goal_state, num_humans, d_safe)

    mppi_params = {
        "prediction_horizon": 10,  # 2 seconds
        "num_samples": 2000 if not os.getenv("CBFKIT_TEST_MODE") else 100,
        "time_step": dt,
        "use_GPU": True,
        "robot_state_dim": 4 + 4 * num_humans,  # Full augmented state
        "robot_control_dim": 2,
        "costs_lambda": 0.1,  # Temp
        "gamma": 0.01,  # Control cost weight in mppi generator (often implicitly handled)
        "cost_perturbation": 0.1,
    }

    planner = mppi_generator()(
        control_limits=jnp.array([robot_dyn.a_max, robot_dyn.omega_max]),
        dynamics_func=aug_dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        mppi_args=mppi_params,
    )

    # Initialize MPPI with a forward-biased trajectory (positive acceleration, zero angular velocity)
    # This warm-starts the optimizer toward forward motion instead of exploring backward
    init_u_traj = jnp.zeros((mppi_params["prediction_horizon"], 2))
    # Set initial acceleration to a moderate positive value to bias toward forward motion
    init_u_traj = init_u_traj.at[:, 0].set(2.0)  # Positive acceleration along entire horizon
    init_planner_data = PlannerData(u_traj=init_u_traj)

    # --- CBF Controller ---
    print("Initializing CBF Safety Filter...")
    barriers = []
    for i in range(num_humans):
        h = create_human_barrier(i, d_safe)
        # Rectify relative degree on augmented dynamics
        # Robot is acceleration controlled -> Rel Deg 2
        # Augmented dynamics handles human velocity naturally
        b = rectify_relative_degree(
            function=h, system_dynamics=aug_dynamics, state_dim=len(z0), form="high-order"
        )(
            certificate_conditions=zeroing_barriers.linear_class_k(
                10.0
            ),  # Very low gain for minimal interference when far from obstacles
        )
        barriers.append(b)

    barrier_package = concatenate_certificates(*barriers)

    controller = cbf_controller(
        control_limits=jnp.array([robot_dyn.a_max, robot_dyn.omega_max]),
        dynamics_func=aug_dynamics,
        barriers=barrier_package,
    )

    # --- Execution ---
    print("Starting Simulation...")
    start_time = time.time()

    x, u, z_sim, p, c_keys, c_values, p_keys, p_values = sim.execute(
        x0=z0,
        dt=dt,
        num_steps=int(tf / dt),
        dynamics=aug_dynamics,
        integrator=integrator,
        planner=planner,
        planner_data=init_planner_data,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        filepath="examples/differential_drive/human_aware_navigation/results/human_aware_results",
        verbose=True,
        use_jit=True,
    )

    print(f"Simulation Complete in {time.time() - start_time:.2f}s")

    return x, u, goal_state, d_safe, num_humans, dt, p_keys, p_values


# --- Visualization ---


def analyze_safety(states, num_humans, d_safe):
    print("\nSafety Analysis:")
    min_dists = []
    collisions = 0

    # Convert to numpy
    states = np.array(states)

    for i in range(len(states)):
        z = states[i]
        p_r = z[:2]
        step_min = float("inf")

        for h in range(num_humans):
            idx = 4 + h * 4
            p_h = z[idx : idx + 2]
            dist = np.linalg.norm(p_r - p_h)
            step_min = min(step_min, dist)
            if dist < d_safe:
                collisions += 1

        min_dists.append(step_min)

    print(f"Minimum Distance Recorded: {min(min_dists):.3f}m")
    print(f"Safety Threshold: {d_safe}m")
    print(f"Total Violation Steps: {collisions}")
    return min(min_dists)


from matplotlib.collections import LineCollection


def create_animation(states, goal_state, num_humans, d_safe, dt, p_keys, p_values):
    print("Generating Animation...")

    # Convert to numpy
    states = np.array(states)
    goal_state = np.array(goal_state)
    p_values = [np.array(val) for val in p_values]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Robot
    rx = states[:, 0]
    ry = states[:, 1]

    # Humans
    hx = []
    hy = []
    hvx = []
    hvy = []
    for h in range(num_humans):
        idx = 4 + h * 4
        hx.append(states[:, idx])
        hy.append(states[:, idx + 1])
        hvx.append(states[:, idx + 2])
        hvy.append(states[:, idx + 3])

    # Limits
    all_x = np.concatenate([rx] + hx + [np.array([goal_state[0]])])
    all_y = np.concatenate([ry] + hy + [np.array([goal_state[1]])])
    margin = 2.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Patches
    robot_patch = Circle((0, 0), 0.3, color="blue", label="Robot")
    goal_patch = Circle((goal_state[0], goal_state[1]), 0.3, color="green", alpha=0.5, label="Goal")

    human_patches = []
    colors = ["red", "orange", "purple"]
    for i in range(num_humans):
        c = Circle(
            (0, 0), d_safe, color=colors[i % len(colors)], alpha=0.2, label=f"Human {i+1} Zone"
        )
        # Core
        cc = Circle((0, 0), 0.2, color=colors[i % len(colors)], alpha=0.8)
        human_patches.append((c, cc))
        ax.add_patch(c)
        ax.add_patch(cc)

    ax.add_patch(robot_patch)
    ax.add_patch(goal_patch)

    (robot_trail,) = ax.plot([], [], "b-", alpha=0.5)
    human_trails = [
        ax.plot([], [], "--", color=colors[i % len(colors)], alpha=0.3)[0]
        for i in range(num_humans)
    ]

    # --- Prediction Visuals (LineCollection) ---

    # MPPI Trajectory Collection
    # We initialize with empty data
    mppi_lc = LineCollection([], linewidths=3, colors="green")
    ax.add_collection(mppi_lc)

    # Human Prediction Collections
    human_lcs = []
    for i in range(num_humans):
        lc = LineCollection([], linewidths=2, linestyles="dotted", colors=colors[i % len(colors)])
        ax.add_collection(lc)
        human_lcs.append(lc)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Extract MPPI data if available
    mppi_traj_idx = -1
    if "x_traj" in p_keys:
        mppi_traj_idx = p_keys.index("x_traj")

    prediction_horizon = 20  # From setup

    def get_fading_segments(x, y):
        """Helper to create segments and alphas for fading line."""
        if len(x) < 2:
            return [], []
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Alphas: 1.0 at start, 0.0 at end
        alphas = np.linspace(1.0, 0.0, len(segments))
        return segments, alphas

    def animate(i):
        t = i * dt

        # Robot
        robot_patch.center = (rx[i], ry[i])
        robot_trail.set_data(rx[:i], ry[:i])

        # MPPI Plan
        if mppi_traj_idx >= 0 and i < len(p_values[mppi_traj_idx]):
            # Shape: (StateDim, Horizon+1)
            traj = p_values[mppi_traj_idx][i]

            # Indices 0 and 1 are Robot X and Y
            # We need to transpose or just slice correct rows
            traj_x = traj[0, :]
            traj_y = traj[1, :]

            segs, alphas = get_fading_segments(traj_x, traj_y)
            mppi_lc.set_segments(segs)
            mppi_lc.set_alpha(alphas)
        else:
            mppi_lc.set_segments([])
        # Humans
        for h in range(num_humans):
            safe_zone, core = human_patches[h]
            pos = (hx[h][i], hy[h][i])
            vel = (hvx[h][i], hvy[h][i])

            safe_zone.center = pos
            core.center = pos
            human_trails[h].set_data(hx[h][:i], hy[h][:i])

            # Prediction (Linear)
            pred_x = [pos[0] + vel[0] * dt * k for k in range(prediction_horizon)]
            pred_y = [pos[1] + vel[1] * dt * k for k in range(prediction_horizon)]

            segs, alphas = get_fading_segments(pred_x, pred_y)
            human_lcs[h].set_segments(segs)
            human_lcs[h].set_alpha(alphas)

        time_text.set_text(f"Time: {t:.1f}s")
        return (
            [robot_patch, robot_trail, mppi_lc, time_text]
            + [p for pair in human_patches for p in pair]
            + human_trails
            + human_lcs
        )

    anim = animation.FuncAnimation(fig, animate, frames=len(states), interval=50, blit=True)

    from cbfkit.utils.animator import save_animation

    save_path = (
        "examples/differential_drive/human_aware_navigation/results/human_aware_animation.mp4"
    )
    save_animation(anim, save_path)

    plt.close()


def main():
    os.makedirs("examples/differential_drive/results", exist_ok=True)
    x, u, goal, d_safe, num_humans, dt, p_keys, p_values = run_simulation()
    analyze_safety(x, num_humans, d_safe)
    if not os.getenv("CBFKIT_TEST_MODE"):
        create_animation(x, goal, num_humans, d_safe, dt, p_keys, p_values)


if __name__ == "__main__":
    main()
