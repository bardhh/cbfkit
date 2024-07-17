'''
In this example, rather than using auto-generation scripts, we pass the stage cost and terminal cost functions directly
'''

import os
from jax import Array, jit
import jax.numpy as jnp
from cbfkit.codegen.create_new_system import generate_model

file_path = os.path.dirname(os.path.abspath(__file__))
target_directory = file_path + "/tutorials"
model_name = "mppi_cbf_stl_v2"

# Simulation Parameters
SAVE_FILE = target_directory + f"/{model_name}/simulation_data"
DT = 0.1  # 1e-2
TF = 10 
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([0.0, 0.0])
ACTUATION_LIMITS = jnp.array([5.0, 5.0])  # Box control input constraint, i.e., -100 <= u <= 100


# Initialize params dict for all dynamics and constraint function related parameters
params = {}

# Define  Dynamics
drift_dynamics = "[0, 0]"
control_matrix = "[[1, 0], [0, 1]]"

# Define functions
goal = jnp.array([4, 4])
goals = [goal] 
goal_threshold = 0.5
goal_radius = 0.5

obstacle = jnp.array([3, 3])
obstacle_radius = 0.6  # 1.0  # 0.6

# Run script for automated genration of dynamics, cost functions python files
generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    # stage_cost_function=stage_cost_function,  # We will pass python function directly below. No need to use auto-generation scripts for that
    # terminal_cost_function=terminal_cost_function,
    params=params,
)

import cbfkit.simulation.simulator as sim
import cbfkit.controllers_and_planners.model_based.mppi as mppi_planner
import cbfkit.controllers_and_planners.waypoint as single_waypoint_planner
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator
from cbfkit.utils.numerical_integration import forward_euler as integrator

from tutorials import mppi_cbf_stl
dynamics = mppi_cbf_stl.plant()

@jit
def stage_cost(state_and_time: Array, action: Array) -> Array:
    """Function to be evaluated.
    Args: state_and_time (Array): concatenated state vector and time
    Returns: Array: cbf value
    """
    x = state_and_time
    return 0.2 * ((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2) + 10.0 / (
        jnp.max(jnp.array([jnp.linalg.norm(x[0:2] - obstacle[0:2]) - obstacle_radius, 0.01]))
    )


@jit
def terminal_cost(state_and_time: Array, action: Array) -> Array:
    """Function to be evaluated.
    Args: state_and_time (Array): concatenated state vector and time
    Returns: Array: cbf value
    """
    x = state_and_time
    return 0.2 * ((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2) + 10.0 / (
        jnp.max(jnp.array([jnp.linalg.norm(x[0:2] - obstacle[0:2]) - obstacle_radius, 0.01]))
    )


# MPPI specific parameters
mppi_args = {
    "robot_state_dim": 2,
    "robot_control_dim": 2,
    "prediction_horizon": 50,  # 100,  # 150,
    "num_samples": 10000,
    "plot_samples": 30,
    "time_step": DT,
    "use_GPU": False,
    "costs_lambda": 0.03,
    "cost_perturbation": 0.1,
}

# Instantiate MPPI control law
mppi_local_planner = mppi_planner.vanilla_mppi(
    control_limits=ACTUATION_LIMITS,
    dynamics_func=dynamics,
    trajectory_cost=None, 
    stage_cost=stage_cost,  
    terminal_cost=terminal_cost,  
    mppi_args=mppi_args,
)

(
    x_,
    u_,
    z_,
    p_,
    controller_data_keys_,
    controller_data_items_,
    planner_data_keys_,
    planner_data_items_,
) = sim.execute(
    x0=INITIAL_STATE,
    dt=DT,
    num_steps=N_STEPS,
    dynamics=dynamics,
    integrator=integrator,
    planner=mppi_local_planner,  # target_setpoint
    nominal_controller=None,  # nominal_controller,
    controller=None,  # cbf_clf_controller,
    sensor=sensor,
    estimator=estimator,
    filepath=SAVE_FILE,
    planner_data={
        "u_traj": 1 * jnp.ones((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"])),
        "prev_robustness": jnp.array([1, -1, -1]),
    },
    controller_data={},
)

plot = True
if plot:
    from tutorials.plot_helper.plot_mppi_ffmpeg import animate

    animate(
        states=x_,
        estimates=z_,
        controller_data_keys=planner_data_keys_,
        controller_data_items=planner_data_items_,
        mppi_args=mppi_args,
        desired_state=goals,
        desired_state_radius=goal_radius + 0.32,
        x_lim=(0, 10),
        y_lim=(0, 10),
        dt=DT,
        title="System Behavior",
        save_animation=True,
        animation_filename=target_directory + "/" + model_name + "/animation",  # + ".mp4",
        obstacle=obstacle,
        obstacle_radius=obstacle_radius,
    )
