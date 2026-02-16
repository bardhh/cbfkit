"""
Pedestrian Head-On Interaction Demo.

Demonstrates the robot navigating a head-on encounter with a pedestrian.
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
    print("Initializing Head-On Demo...")

    manager = CrowdManager()

    # Pedestrian moving Left towards robot
    manager.add_pedestrian(
        init_state=jnp.array([10.0, 5.0, -1.0, 0.0]),
        behavior=social_force_policy(goal=jnp.array([0.0, 5.0]), desired_speed=1.0),
        id="ped_head_on",
    )

    num_peds = len(manager.pedestrians)

    # Robot moving Right towards pedestrian
    robot_dyn = unicycle.plant(l=1.0)
    robot_dyn.a_max = 5.0
    robot_dyn.omega_max = 5.0
    robot_dyn.v_max = 4.0

    x0_robot = jnp.array([0.0, 5.0, 0.0, 0.0])
    goal_robot = jnp.array([10.0, 5.0, 0.0, 0.0])

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
            c_obs += 150.0 * jnp.exp(-2.5 * (dist_h - 1.0))  # Stronger repulsion for head-on

        return 0.01 * jnp.dot(u, u) + 2.0 * dist_goal + c_obs

    @jit
    def terminal_cost(z):
        p_r = z[:2]
        return 10.0 * jnp.linalg.norm(p_r - goal_robot[:2]) ** 2

    mppi_params = {
        "prediction_horizon": 25,
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

    init_planner_data = PlannerData(u_traj=jnp.zeros((25, 2)))
    init_controller_data = ControllerData(sub_data={"inner_controller_data": init_planner_data})

    combined_controller = manager.get_nominal_controller(planner, use_augmented_state=True)

    dt = 0.1
    if os.getenv("CBFKIT_TEST_MODE"):
        tf = 1.0
    else:
        tf = 10.0

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
    if not os.getenv("CBFKIT_TEST_MODE"):
        visualize_crowd(
            states=x,
            num_pedestrians=num_peds,
            robot_goal=goal_robot,
            d_safe=1.0,
            dt=dt,
            p_values=p_values,
            p_keys=p_keys,
            save_path=str(results_dir / "head_on.mp4"),
        )

    print("Demo Complete!")


if __name__ == "__main__":
    run_demo()
