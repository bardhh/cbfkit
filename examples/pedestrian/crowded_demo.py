"""
Crowded Environment Navigation Demo.

Demonstrates the robot navigating through a busy environment with multiple pedestrians moving in a naturalistic manner using the Social Force Model.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

# Add project root
sys.path.append(os.getcwd())

from jax import jit

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.controllers.mppi.mppi_generator import mppi_generator
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems.pedestrian import CrowdManager
from cbfkit.systems.pedestrian.behaviors import social_force_policy
from cbfkit.utils.user_types import ControllerData, PlannerData
from cbfkit.utils.visualization import visualize_crowd


def run_demo():
    print("Initializing Naturalistic Crowded Demo...")

    manager = CrowdManager()

    # Create a more naturalistic scenario with pedestrians moving towards different goals
    # mimicking a busy plaza or intersection.

    # Pedestrian 1: Bottom-Left -> Top-Right (Diagonal)
    manager.add_pedestrian(
        init_state=[2.0, 2.0, 0.5, 0.5],
        behavior=social_force_policy(goal=jnp.array([12.0, 12.0]), desired_speed=1.0),
        id="ped_diag_1",
    )
    # Pedestrian 2: Top-Left -> Bottom-Right (Diagonal, crossing P1)
    manager.add_pedestrian(
        init_state=[2.0, 12.0, 0.5, -0.5],
        behavior=social_force_policy(goal=jnp.array([12.0, 2.0]), desired_speed=1.1),
        id="ped_diag_2",
    )
    # Pedestrian 3: Right -> Left (Horizontal, slightly curved path due to interactions)
    manager.add_pedestrian(
        init_state=[12.0, 7.0, -1.0, 0.0],
        behavior=social_force_policy(goal=jnp.array([0.0, 7.0]), desired_speed=0.9),
        id="ped_horiz",
    )
    # Pedestrian 4: Bottom -> Top (Vertical, faster)
    manager.add_pedestrian(
        init_state=[7.0, 0.0, 0.0, 1.2],
        behavior=social_force_policy(goal=jnp.array([7.0, 14.0]), desired_speed=1.3),
        id="ped_vert",
    )
    # Pedestrian 5: Loitering/Slow moving (Top-Right area)
    manager.add_pedestrian(
        init_state=[10.0, 10.0, -0.2, -0.2],
        behavior=social_force_policy(goal=jnp.array([8.0, 8.0]), desired_speed=0.5),
        id="ped_slow",
    )

    num_peds = len(manager.pedestrians)

    # Robot Setup (Start Left, Goal Right)
    robot_dyn = unicycle.plant(l=1.0)
    robot_dyn.a_max = 5.0
    robot_dyn.omega_max = 5.0
    robot_dyn.v_max = 4.0

    x0_robot = jnp.array([0.0, 5.0, 0.0, 0.0])
    goal_robot = jnp.array([14.0, 8.0, 0.0, 0.0])

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
            # Soft repulsion field
            c_obs += 150.0 * jnp.exp(-2.5 * (dist_h - 0.8))

        return 0.01 * jnp.dot(u, u) + 2.0 * dist_goal + c_obs

    @jit
    def terminal_cost(z):
        p_r = z[:2]
        return 10.0 * jnp.linalg.norm(p_r - goal_robot[:2]) ** 2

    mppi_params = {
        "prediction_horizon": 30,
        "num_samples": 1000,  # High samples for complex interactions
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
    tf = 15.0  # Allow time for complex navigation

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

    os.makedirs("examples/pedestrian/results", exist_ok=True)
    visualize_crowd(
        states=x,
        num_pedestrians=num_peds,
        robot_goal=goal_robot,
        d_safe=0.8,
        dt=dt,
        p_values=p_values,
        p_keys=p_keys,
        save_path="examples/pedestrian/results/crowded_naturalistic_demo.mp4",
    )

    print("Demo Complete!")


if __name__ == "__main__":
    run_demo()
