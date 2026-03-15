"""
Multi-Scenario Human-Aware Navigation.

Generates a 4-quadrant visualization of a robot navigating among humans
in different scenarios using MPPI + CBF.

Scenarios:
1. Crossing: Two humans crossing the robot's path perpendicularly.
2. Crowded: Four humans in a dense configuration.
3. Head-On: A human walking directly towards the robot.
4. Overtaking: Humans moving in the same direction at different speeds.
"""

import os
import sys
import time

# Add the project root directory to the python path
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
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller
from cbfkit.controllers.mppi import vanilla_mppi
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.utils.user_types import PlannerData

# --- Shared Helper Functions ---


def human_cv_dynamics():
    def f(x):
        return jnp.array([x[2], x[3], 0.0, 0.0])

    return f


def create_augmented_dynamics(robot_dyn, num_humans):
    human_f = human_cv_dynamics()

    def dynamics(z):
        x_r = z[:4]
        f_r, g_r = robot_dyn(x_r)
        f_list = [f_r]
        g_rows = [g_r]
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


def create_mppi_costs(goal_state, num_humans, d_safe):
    R_val = 0.01
    Q_goal = 2.0
    Q_obs = 100.0

    @jit
    def stage_cost(z, u):
        c_u = R_val * jnp.dot(u, u)
        p_r = z[:2]
        p_g = goal_state[:2]
        dist_goal = jnp.linalg.norm(p_r - p_g)
        c_goal = Q_goal * dist_goal
        c_obs = 0.0
        for i in range(num_humans):
            idx = 4 + i * 4
            p_h = z[idx : idx + 2]
            dist_h = jnp.linalg.norm(p_r - p_h)
            c_obs += Q_obs * jnp.exp(-2.0 * (dist_h - d_safe))
        return c_u + c_goal + c_obs

    @jit
    def terminal_cost(z):
        p_r = z[:2]
        p_g = goal_state[:2]
        return 10.0 * jnp.linalg.norm(p_r - p_g) ** 2

    return stage_cost, terminal_cost


def create_human_barrier(human_idx, d_min):
    idx = 4 + human_idx * 4

    def h(z):
        p_r = z[:2]
        p_h = z[idx : idx + 2]
        diff = p_r - p_h
        return jnp.dot(diff, diff) - d_min**2

    return h


# --- Simulation Runner ---


def run_scenario(config):
    print(f"Running Scenario: {config['name']}")

    # Unpack Config
    x0_r = jnp.array(config["robot_init"])
    goal = jnp.array(config["robot_goal"])
    humans = config["humans"]
    num_humans = len(humans)
    d_safe = config.get("d_safe", 1.0)
    dt = config.get("dt", 0.1)
    tf = config.get("tf", 10.0)

    # Dynamics
    robot_dyn = unicycle.plant(l=1.0)
    robot_dyn.a_max = config.get("a_max", 5.0)
    robot_dyn.omega_max = config.get("omega_max", 5.0)
    robot_dyn.v_max = config.get("v_max", 4.0)

    aug_dynamics = create_augmented_dynamics(robot_dyn, num_humans)

    # Initial State
    human_states = [jnp.array(h["init"]) for h in humans]
    z0 = jnp.concatenate([x0_r] + human_states)

    # MPPI
    stage_cost, terminal_cost = create_mppi_costs(goal, num_humans, d_safe)
    mppi_params = {
        "prediction_horizon": 20,
        "num_samples": 1000,
        "time_step": dt,
        "use_GPU": True,
        "robot_state_dim": 4 + 4 * num_humans,
        "robot_control_dim": 2,
        "costs_lambda": 0.1,
        "gamma": 0.01,
        "cost_perturbation": 0.1,
    }

    planner = vanilla_mppi(
        control_limits=jnp.array([robot_dyn.a_max, robot_dyn.omega_max]),
        dynamics_func=aug_dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        mppi_args=mppi_params,
    )
    init_planner_data = PlannerData(u_traj=jnp.zeros((mppi_params["prediction_horizon"], 2)))

    # CBF
    barriers = []
    for i in range(num_humans):
        h = create_human_barrier(i, d_safe)
        b = rectify_relative_degree(
            function=h, system_dynamics=aug_dynamics, state_dim=len(z0), form="high-order"
        )(
            certificate_conditions=zeroing_barriers.linear_class_k(1.0),
        )
        barriers.append(b)

    if barriers:
        barrier_package = concatenate_certificates(*barriers)
        controller = cbf_controller(
            control_limits=jnp.array([robot_dyn.a_max, robot_dyn.omega_max]),
            dynamics_func=aug_dynamics,
            barriers=barrier_package,
        )
    else:
        # Fallback if no humans (not used in this script but good practice)
        # Just pass a dummy controller or handle separately.
        # For now assuming always > 0 humans.
        barrier_package = concatenate_certificates(*barriers)
        controller = cbf_controller(
            control_limits=jnp.array([robot_dyn.a_max, robot_dyn.omega_max]),
            dynamics_func=aug_dynamics,
            barriers=barrier_package,
        )

    # Execute
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
        verbose=False,  # Reduce noise
    )

    return {
        "config": config,
        "states": x,
        "p_keys": p_keys,
        "p_values": p_values,
        "dt": dt,
        "num_humans": num_humans,
        "d_safe": d_safe,
    }


# --- Scenarios Definition ---


def define_scenarios():
    scenarios = []

    if os.getenv("CBFKIT_TEST_MODE"):
        scenarios.append(
            {
                "name": "Crossing",
                "robot_init": [0.0, 5.0, 0.0, 0.0],
                "robot_goal": [10.0, 5.0, 0.0, 0.0],
                "humans": [
                    {"init": [5.0, 0.0, 0.0, 1.0]},  # Up
                    {"init": [5.0, 10.0, 0.0, -1.0]},  # Down
                ],
                "d_safe": 1.0,
                "tf": 1.0,
            }
        )
        return scenarios

    # 1. Crossing
    scenarios.append(
        {
            "name": "Crossing",
            "robot_init": [0.0, 5.0, 0.0, 0.0],
            "robot_goal": [10.0, 5.0, 0.0, 0.0],
            "humans": [
                {"init": [5.0, 0.0, 0.0, 1.0]},  # Up
                {"init": [5.0, 10.0, 0.0, -1.0]},  # Down
            ],
            "d_safe": 1.0,
        }
    )

    # 2. Crowded (Grid)
    scenarios.append(
        {
            "name": "Crowded",
            "robot_init": [0.0, 0.0, 0.78, 0.0],  # Diagonal start
            "robot_goal": [10.0, 10.0, 0.0, 0.0],
            "humans": [
                {"init": [3.0, 0.0, 0.0, 1.0]},
                {"init": [0.0, 3.0, 1.0, 0.0]},
                {"init": [7.0, 10.0, 0.0, -1.0]},
                {"init": [10.0, 7.0, -1.0, 0.0]},
            ],
            "d_safe": 0.8,
        }
    )

    # 3. Head-On
    scenarios.append(
        {
            "name": "Head-On",
            "robot_init": [1.0, 5.0, 0.0, 1.0],
            "robot_goal": [9.0, 5.0, 0.0, 0.0],
            "humans": [
                {"init": [9.0, 5.0, -1.2, 0.0]},  # Moving towards robot
            ],
            "d_safe": 1.2,
        }
    )

    # 4. Overtaking
    scenarios.append(
        {
            "name": "Overtaking",
            "robot_init": [0.0, 5.0, 0.0, 2.0],  # Faster start
            "robot_goal": [15.0, 5.0, 0.0, 0.0],
            "humans": [
                {"init": [4.0, 5.0, 1.0, 0.0]},  # Slower, same dir
                {"init": [8.0, 5.5, 0.8, 0.0]},  # Slower, slight offset
            ],
            "d_safe": 1.0,
            "tf": 12.0,
            "v_max": 6.0,  # Robot can go faster
        }
    )

    return scenarios


# --- Visualization ---


def create_multi_animation(results):
    print("Generating Combined Animation...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Storage for artists per subplot
    plot_data = []

    for idx, res in enumerate(results):
        ax = axes[idx]
        config = res["config"]
        states = res["states"]
        p_keys = res["p_keys"]
        p_values = res["p_values"]
        num_humans = res["num_humans"]
        d_safe = res["d_safe"]
        dt = res["dt"]
        goal = config["robot_goal"]

        ax.set_title(f"{config['name']}", fontsize=14, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Calculate limits
        rx = states[:, 0]
        ry = states[:, 1]
        all_x = list(rx) + [goal[0]]
        all_y = list(ry) + [goal[1]]
        for h in range(num_humans):
            hx = states[:, 4 + h * 4]
            hy = states[:, 4 + h * 4 + 1]
            all_x.extend(hx)
            all_y.extend(hy)

        margin = 1.5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        # Static elements
        ax.add_patch(Circle((goal[0], goal[1]), 0.4, color="green", alpha=0.5, label="Goal"))

        # Dynamic elements
        robot_patch = Circle((0, 0), 0.3, color="blue", label="Robot")
        ax.add_patch(robot_patch)
        (robot_trail,) = ax.plot([], [], "b-", alpha=0.5)

        human_patches = []
        human_trails = []
        human_preds = []  # Prediction lines
        human_colors = ["red", "orange", "purple", "brown"]

        for h in range(num_humans):
            c = Circle((0, 0), d_safe, color=human_colors[h % 4], alpha=0.15)
            cc = Circle((0, 0), 0.2, color=human_colors[h % 4], alpha=0.8)
            ax.add_patch(c)
            ax.add_patch(cc)
            human_patches.append((c, cc))

            (tr,) = ax.plot([], [], "--", color=human_colors[h % 4], alpha=0.3)
            human_trails.append(tr)

            # Human Prediction LineCollection
            lc = LineCollection([], linewidths=1.5, linestyles="dotted", colors=human_colors[h % 4])
            ax.add_collection(lc)
            human_preds.append(lc)

        # MPPI LineCollection
        mppi_lc = LineCollection([], linewidths=2.5, colors="green", alpha=0.8)
        ax.add_collection(mppi_lc)

        # Find MPPI index
        mppi_idx = -1
        if "x_traj" in p_keys:
            mppi_idx = p_keys.index("x_traj")

        plot_data.append(
            {
                "ax": ax,
                "states": states,
                "robot_patch": robot_patch,
                "robot_trail": robot_trail,
                "human_patches": human_patches,
                "human_trails": human_trails,
                "human_preds": human_preds,
                "mppi_lc": mppi_lc,
                "mppi_idx": mppi_idx,
                "p_values": p_values,
                "dt": dt,
                "num_humans": num_humans,
            }
        )

    # Common helper for fading
    def get_fading_segments(x, y):
        if len(x) < 2:
            return [], []
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(1.0, 0.0, len(segments))
        return segments, alphas

    max_frames = max(len(r["states"]) for r in results)

    def animate(frame):
        artists = []

        for data in plot_data:
            states = data["states"]
            if frame >= len(states):
                current_frame = len(states) - 1
            else:
                current_frame = frame

            # Robot
            rx, ry = states[current_frame, 0], states[current_frame, 1]
            data["robot_patch"].center = (rx, ry)
            data["robot_trail"].set_data(states[:current_frame, 0], states[:current_frame, 1])
            artists.extend([data["robot_patch"], data["robot_trail"]])

            # MPPI
            mppi_idx = data["mppi_idx"]
            if mppi_idx >= 0 and current_frame < len(data["p_values"][mppi_idx]):
                traj = data["p_values"][mppi_idx][current_frame]
                # Traj shape (StateDim, Horizon+1) -> (0, :) is X, (1, :) is Y
                segs, alphas = get_fading_segments(traj[0, :], traj[1, :])
                data["mppi_lc"].set_segments(segs)
                data["mppi_lc"].set_alpha(alphas)
            else:
                data["mppi_lc"].set_segments([])
            artists.append(data["mppi_lc"])

            # Humans
            for h in range(data["num_humans"]):
                zone, core = data["human_patches"][h]
                idx = 4 + h * 4
                hx, hy = states[current_frame, idx], states[current_frame, idx + 1]
                hvx, hvy = states[current_frame, idx + 2], states[current_frame, idx + 3]

                zone.center = (hx, hy)
                core.center = (hx, hy)
                data["human_trails"][h].set_data(
                    states[:current_frame, idx], states[:current_frame, idx + 1]
                )

                # Prediction
                pred_x = [hx + hvx * data["dt"] * k for k in range(20)]
                pred_y = [hy + hvy * data["dt"] * k for k in range(20)]
                segs, alphas = get_fading_segments(pred_x, pred_y)
                data["human_preds"][h].set_segments(segs)
                data["human_preds"][h].set_alpha(alphas)

                artists.extend([zone, core, data["human_trails"][h], data["human_preds"][h]])

        return artists

    anim = animation.FuncAnimation(fig, animate, frames=max_frames, interval=50, blit=True)

    save_path = os.path.abspath("examples/differential_drive/human_aware_navigation/results/multi_scenario_animation.mp4")
    try:
        anim.save(save_path, writer="ffmpeg", fps=20)
        print(f"\nAnimation saved to: file://{save_path}")
    except Exception:
        save_path_gif = save_path.replace("mp4", "gif")
        anim.save(save_path_gif, writer="pillow", fps=20)
        print(f"\nAnimation saved to: file://{save_path_gif}")

    plt.close()


def main():
    os.makedirs("examples/differential_drive/results", exist_ok=True)

    scenarios = define_scenarios()
    results = []

    for sc in scenarios:
        res = run_scenario(sc)
        results.append(res)

    if not os.getenv("CBFKIT_TEST_MODE"):
        create_multi_animation(results)


if __name__ == "__main__":
    main()
