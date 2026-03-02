"""
Pedestrian Manager Demo.

Demonstrates the use of `cbfkit.systems.pedestrian.CrowdManager` to easily setup
a robot navigation scenario with multiple pedestrians using Social Force behavior.

Scenario: 'Crossing'
- Robot starts at (0, 5) heading Right.
- 2 Pedestrians cross vertically.
"""

import os
import sys

import jax.numpy as jnp

# Add project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from jax import jit

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.controllers.mppi.mppi_generator import mppi_generator
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor

# --- New Imports ---
from cbfkit.systems.pedestrian import CrowdManager
from cbfkit.systems.pedestrian.behaviors import social_force_policy
from cbfkit.utils.user_types import ControllerData, PlannerData
from cbfkit.utils.visualization import visualize_crowd


def run_demo():
    print("Initializing CrowdManager Demo...")

    # 1. Setup Manager
    manager = CrowdManager()

    # 2. Add Pedestrians (Social Force)
    # Pedestrian 1: Moving Up
    manager.add_pedestrian(
        init_state=[5.0, 0.0, 0.0, 0],
        behavior=social_force_policy(goal=jnp.array([0.0, 0.0]), desired_speed=1.0),
    )
    # Pedestrian 2: Moving Down
    manager.add_pedestrian(
        init_state=[10.0, 6.0, 0.0, 0],
        behavior=social_force_policy(goal=jnp.array([0.0, 0.0]), desired_speed=1.0),
    )

    num_peds = len(manager.pedestrians)

    # 3. Robot Setup
    robot_dyn = unicycle.plant(l=1.0)
    robot_dyn.a_max = 5.0
    robot_dyn.omega_max = 5.0
    robot_dyn.v_max = 4.0

    x0_robot = jnp.array([0.0, 0.0, 0.0, 0.0])
    goal_robot = jnp.array([10.0, 5.0, 0.0, 0.0])

    # 4. Get Augmented System
    aug_dynamics = manager.get_augmented_dynamics(robot_dyn)
    closed_loop_dynamics = manager.get_closed_loop_dynamics(robot_dyn)
    z0 = manager.get_initial_state(x0_robot)

    # 5. Configure MPPI (Nominal Planner)
    # MPPI Costs
    @jit
    def stage_cost(z, u):
        p_r = z[:2]
        dist_goal = jnp.linalg.norm(p_r - goal_robot[:2])

        # Simple Obstacle Repulsion (based on augmented state z)
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
        "prediction_horizon": 20,
        "num_samples": 500,
        "time_step": 0.1,
        "use_GPU": True,
        "robot_state_dim": len(z0),
        "robot_control_dim": 2,
        "costs_lambda": 0.1,
        "gamma": 0.01,
        "cost_perturbation": 0.1,
    }

    # Use closed_loop_dynamics for planning
    planner = mppi_generator()(
        control_limits=jnp.array([5.0, 5.0]),
        dynamics_func=closed_loop_dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        mppi_args=mppi_params,
    )

    init_planner_data = PlannerData(u_traj=jnp.zeros((20, 2)))

    # Wrap planner data in ControllerData because we use MPPI as controller
    # Use 'inner_controller_data' key for CrowdManager to unpack
    init_controller_data = ControllerData(sub_data={"inner_controller_data": init_planner_data})

    # 6. Wrap Controller
    combined_controller = manager.get_nominal_controller(planner, use_augmented_state=True)

    # 7. CBF Safety Filter (Optional but Recommended)
    # For simplicity in this demo, we might skip CBF or add it.
    # If we add it, we must wrap the CBF controller which wraps the Nominal Controller.
    # Let's stick to MPPI only for this demo to verify the CrowdManager logic first.
    # If MPPI avoids, the dynamics/manager integration is correct.

    # 8. Run Simulation
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
        # Note: We pass 'combined_controller' as the primary 'controller' here if we don't use CBF.
        # If we used CBF, we'd pass CBF as controller and combined_controller as nominal.
        # But `sim.execute` logic calls `controller`.
        controller=combined_controller,
        sensor=sensor,
        estimator=estimator,
        controller_data=init_controller_data,  # Pass wrapped data
        verbose=True,
        use_jit=True,
    )

    # 9. Visualize
    # Unpack p_values from controller_data if needed?
    # sim.execute returns p_keys, p_values if planner is used.
    # Here planner is NONE. So p_keys/values will be empty.
    # The planner data is hidden inside c_values (ControllerData).
    # c_keys will contain 'sub_data' maybe?
    # Actually, sim.execute logs controller_data fields.

    # We need to extract MPPI trajectories for visualization from c_values.
    # This might be tricky if c_values flattens sub_data.
    # Let's see what we get.

    os.makedirs("examples/pedestrian/results", exist_ok=True)
    if not os.getenv("CBFKIT_TEST_MODE"):
        visualize_crowd(
            states=x,
            num_pedestrians=num_peds,
            robot_goal=goal_robot,
            d_safe=1.0,
            dt=dt,
            # p_values=p_values, # Likely empty
            # p_keys=p_keys,
            save_path="examples/pedestrian/navigate_among_pedestrians/results/crossing.mp4",
        )

    print("Demo Complete!")


if __name__ == "__main__":
    run_demo()
