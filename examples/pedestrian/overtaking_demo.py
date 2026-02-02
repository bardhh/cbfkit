"""
Pedestrian Overtaking Demo.

Demonstrates the robot overtaking slower moving pedestrians.
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

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
    print("Initializing Overtaking Demo...")

    manager = CrowdManager()

    # Robot moves faster (v_max=4.0)
    # Add slower pedestrians moving in the same direction

    # Pedestrian 1: Ahead, moving right, slow
    manager.add_pedestrian(
        init_state=jnp.array([5.0, 5.0, 0.8, 0.0]),
        behavior=social_force_policy(goal=jnp.array([20.0, 5.0]), desired_speed=0.8),
        id="ped_slow_1",
    )
    # Pedestrian 2: Further ahead, slightly offset, moving right, slow
    manager.add_pedestrian(
        init_state=jnp.array([9.0, 4.5, 0.7, 0.0]),
        behavior=social_force_policy(goal=jnp.array([20.0, 4.5]), desired_speed=0.7),
        id="ped_slow_2",
    )

    num_peds = len(manager.pedestrians)

    # Robot Setup
    robot_dyn = unicycle.plant(l=1.0)
    robot_dyn.a_max = 5.0
    robot_dyn.omega_max = 5.0
    robot_dyn.v_max = 4.0

    x0_robot = jnp.array([0.0, 5.0, 0.0, 0.0])
    goal_robot = jnp.array([15.0, 5.0, 0.0, 0.0])

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
            c_obs += 100.0 * jnp.exp(-2.0 * (dist_h - 1.0))

        return 0.01 * jnp.dot(u, u) + 2.0 * dist_goal + c_obs

    @jit
    def terminal_cost(z):
        p_r = z[:2]
        return 10.0 * jnp.linalg.norm(p_r - goal_robot[:2]) ** 2

    mppi_params = {
        "prediction_horizon": 30,
        "num_samples": 500,
        "time_step": 0.1,
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

    init_planner_data = PlannerData(u_traj=jnp.zeros((30, 2)))
    init_controller_data = ControllerData(sub_data={"inner_controller_data": init_planner_data})

    combined_controller = manager.get_nominal_controller(planner, use_augmented_state=True)

    dt = 0.1
    tf = 12.0  # Longer time for overtaking

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
    )

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    visualize_crowd(
        states=x,
        num_pedestrians=num_peds,
        robot_goal=goal_robot,
        d_safe=1.0,
        dt=dt,
        p_values=(
            p_values if len(p_keys) > 0 else None
        ),  # Extract from controller data if needed, usually p_values is returned if planner used directly or via wrapper if sim supports it.
        p_keys=p_keys,
        save_path=str(results_dir / "overtaking_demo.mp4"),
    )

    print("Demo Complete!")


if __name__ == "__main__":
    run_demo()
