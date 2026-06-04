"""MPPI sampling-based reach-avoid planning for a 2D single integrator."""
import os
import sys

# Add the project root to the path so we can import cbfkit + examples.
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import numpy as np
from jax import Array, jit
from matplotlib.animation import FuncAnimation, PillowWriter

import cbfkit.controllers.mppi as mppi_planner
import cbfkit.simulation.simulator as sim
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor

# CBFKIT_TEST_MODE: short horizon and skip the GIF render entirely.
TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))

DT = 0.1
TF = 1.0 if TEST_MODE else 8.0
N_STEPS = int(TF / DT) + 1
x0 = jnp.array([0.0, 0.0])
goal = jnp.array([4.0, 4.0])
obstacle = jnp.array([3.0, 3.0])
obstacle_radius = 0.6

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
GIF_PATH = os.path.join(RESULTS_DIR, "mppi_reach_avoid.gif")


def plant():
    def dynamics(x):
        return jnp.zeros(2), jnp.eye(2)

    return dynamics


dynamics = plant()


@jit
def stage_cost(state_and_time: Array, action: Array) -> Array:
    x = state_and_time
    dist_goal_sq = (x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2
    margin = jnp.maximum(jnp.linalg.norm(x[0:2] - obstacle[0:2]) - obstacle_radius, 0.01)
    return 5.0 * dist_goal_sq + 8.0 / margin + 0.1 * (action[0] ** 2 + action[1] ** 2)


@jit
def terminal_cost(state_and_time: Array, action: Array) -> Array:
    x = state_and_time
    return 50.0 * ((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2)


def main():
    mppi_args = {
        "robot_state_dim": 2,
        "robot_control_dim": 2,
        "prediction_horizon": 25,
        "num_samples": 2000,
        "plot_samples": 30,
        "time_step": DT,
        "use_GPU": False,
        "costs_lambda": 0.03,
        "cost_perturbation": 0.1,
    }
    planner = mppi_planner.vanilla_mppi(
        control_limits=jnp.array([5.0, 5.0]),
        dynamics_func=dynamics,
        trajectory_cost=None,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        mppi_args=mppi_args,
    )

    res = sim.execute(
        x0=x0,
        dt=DT,
        num_steps=N_STEPS,
        dynamics=dynamics,
        integrator=integrator,
        planner=planner,
        nominal_controller=None,
        controller=None,
        sensor=sensor,
        estimator=estimator,
        planner_data={
            "u_traj": jnp.ones((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"])),
        },
        controller_data={},
    )
    states = np.asarray(res["states"])

    final_dist = float(np.linalg.norm(states[-1] - np.asarray(goal)))
    print(f"[mppi_reach_avoid] final distance to goal: {final_dist:.3f}")

    if TEST_MODE:
        # Fast path: skip the GIF render, just report the reach metric.
        print("[mppi_reach_avoid] CBFKIT_TEST_MODE: skipping GIF render.")
        return final_dist

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(
        plt.Circle(
            (float(obstacle[0]), float(obstacle[1])),
            obstacle_radius,
            color="red",
            alpha=0.35,
        )
    )
    ax.plot(float(goal[0]), float(goal[1]), "g*", markersize=18, label="Goal")
    (line,) = ax.plot([], [], "b-", lw=2)
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_aspect("equal")
    ax.set_title("MPPI — sampling-based reach-avoid planning", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(i):
        line.set_data(states[: i + 1, 0], states[: i + 1, 1])
        dot.set_offsets([[states[i, 0], states[i, 1]]])
        return line, dot

    stride = max(1, len(states) // 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    anim.save(GIF_PATH, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"[mppi_reach_avoid] saved GIF -> {GIF_PATH}")
    return final_dist


if __name__ == "__main__":
    main()
