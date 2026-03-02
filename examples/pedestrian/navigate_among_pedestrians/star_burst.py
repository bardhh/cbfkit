"""
Crowded Star Burst Demo.

Demonstrates a highly naturalistic and chaotic scenario where pedestrians
start from a circle and move towards opposing sides (Star Burst pattern),
forcing the robot to navigate through the center of the crowd.
"""

import os
import sys
import time

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from jax import jit

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.controllers.mppi.mppi_generator import mppi_generator
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems.pedestrian import CrowdManager
from cbfkit.systems.pedestrian.behaviors import social_force_policy
from cbfkit.utils.user_types import ControllerData, PlannerData
from cbfkit.utils.visualization import visualize_crowd


def run_demo():
    print("Initializing Star Burst Demo...")

    manager = CrowdManager()

    # Scenario: Circle of pedestrians moving to opposite side
    # Center at (10, 10). Radius 8.
    center = np.array([10.0, 10.0])
    radius = 8.0
    num_peds_circle = 6

    for i in range(num_peds_circle):
        angle = 2 * np.pi * i / num_peds_circle
        start_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
        # Goal is opposite side
        goal_pos = center - radius * np.array([np.cos(angle), np.sin(angle)])

        # Initial velocity towards center
        direction = goal_pos - start_pos
        direction = direction / np.linalg.norm(direction)
        speed = 0.5 + 0.5 * np.random.rand()  # Randomize speed slightly
        init_vel = direction * speed

        manager.add_pedestrian(
            init_state=[start_pos[0], start_pos[1], init_vel[0], init_vel[1]],
            behavior=social_force_policy(
                goal=jnp.array(goal_pos),
                desired_speed=1.2,
                repulsion_strength=5.0,  # Higher repulsion to force avoidance in center
                repulsion_range=0.5,
            ),
            id=f"ped_{i}",
        )

    num_peds = len(manager.pedestrians)

    # Robot Setup: Crosses the circle horizontally
    robot_dyn = unicycle.plant(l=1.0)
    robot_dyn.a_max = 5.0
    robot_dyn.omega_max = 5.0
    robot_dyn.v_max = 4.0

    x0_robot = jnp.array([1.0, 2.0, 0.0, 0.0])  # Start Left
    goal_robot = jnp.array([18.0, 10.0, 0.0, 0.0])  # Goal Right

    # Augmented System
    aug_dynamics = manager.get_augmented_dynamics(robot_dyn)
    closed_loop_dynamics = manager.get_closed_loop_dynamics(robot_dyn)
    z0 = manager.get_initial_state(x0_robot)

    # MPPI Costs
    @jit
    def stage_cost(z, u):
        p_r = z[:2]
        dist_goal = jnp.linalg.norm(p_r - goal_robot[:2])

        c_obs = 0.0
        for i in range(num_peds):
            idx = 4 + i * 4
            p_h = z[idx : idx + 2]
            dist_h = jnp.linalg.norm(p_r - p_h)
            # Stronger soft repulsion for MPPI to anticipate crowding
            c_obs += 200.0 * jnp.exp(-2.0 * (dist_h - 1.0))

        return 0.01 * jnp.dot(u, u) + 2.0 * dist_goal + c_obs

    @jit
    def terminal_cost(z):
        p_r = z[:2]
        return 10.0 * jnp.linalg.norm(p_r - goal_robot[:2]) ** 2

    prediction_horizon = 60
    mppi_params = {
        "prediction_horizon": prediction_horizon,
        "num_samples": 500,
        "time_step": 0.05,
        "use_GPU": True,
        "robot_state_dim": len(z0),
        "robot_control_dim": 2,
        "costs_lambda": 0.1,
        "gamma": 0.01,
        "cost_perturbation": 0.1,
    }

    planner = mppi_generator()(
        control_limits=jnp.array([5.0, 5.0]),
        dynamics_func=closed_loop_dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        mppi_args=mppi_params,
    )

    init_planner_data = PlannerData(u_traj=jnp.zeros((prediction_horizon, 2)))
    init_controller_data = ControllerData(sub_data={"inner_controller_data": init_planner_data})

    combined_controller = manager.get_nominal_controller(planner, use_augmented_state=True)

    dt = 0.05
    if os.getenv("CBFKIT_TEST_MODE"):
        tf = 1.0
    else:
        tf = 20.0

    print("Starting Simulation...")
    x, u, z_sim, p, c_keys, c_values, p_keys, p_values = sim.execute(
        x0=z0,
        dt=dt,
        num_steps=int(tf / dt),
        dynamics=aug_dynamics,
        integrator=integrator,
        controller=combined_controller,
        sensor=sensor,
        estimator=estimator,
        controller_data=init_controller_data,
        verbose=True,
        use_jit=True,
        jit_progress=True,
    )

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    if not os.getenv("CBFKIT_TEST_MODE"):
        visualize_crowd(
            states=x,
            num_pedestrians=num_peds,
            robot_goal=goal_robot,
            d_safe=0.8,
            dt=dt,
            p_values=p_values,
            p_keys=p_keys,
            save_path=os.path.join(results_dir, "star_burst.mp4"),
        )

    print("Demo Complete!")


if __name__ == "__main__":
    run_demo()
