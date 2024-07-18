import os
from jax import Array, jit
import jax.numpy as jnp
from cbfkit.codegen.create_new_system import generate_model

from cbfkit.utils.jax_stl import *

file_path = os.path.dirname(os.path.abspath(__file__))
target_directory = file_path + "/tutorials"
model_name = "mppi_cbf_stl"

# Simulation Parameters
SAVE_FILE = target_directory + f"/{model_name}/simulation_data"
DT = 0.1  # 1e-2
TF = 10  # 0  # 20  # 0  # 10.0
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([0.0, 0.0])
ACTUATION_LIMITS = jnp.array([5.0, 5.0])  # Box control input constraint, i.e., -100 <= u <= 100

goal = jnp.array([7, 7])  # .reshape(-1, 1)
goal_radius = 0.5
obstacle = jnp.array([3, 6])

# Generate files automatically
params = {}
drift_dynamics = "[0, 0]"  # Dynamics
control_matrix = "[[1, 0], [0, 1]]"  # Dynamics
goal = jnp.array([4, 4])
goal2 = jnp.array([6, 9])
goal3 = jnp.array([4, 1])
goals = [goal, goal2, goal3]
obstacle = jnp.array([3, 3])
obstacle_radius = 0.6
goal_threshold = 0.5


generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    # stage_cost_function=stage_cost_function,
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

mppi_args = {
    "robot_state_dim": 2,
    "robot_control_dim": 2,
    "prediction_horizon": 80,
    "num_samples": 10000,
    "plot_samples": 30,
    "time_step": DT,
    "use_GPU": False,
    "costs_lambda": 0.03,
    "cost_perturbation": 0.1,
}

##################################
# Define STL constraint and trajectory cost (based on robustness metric)
# The current implementation requires the trajectory_cost function to be defined two times
# 1. stl_complete_trajectory_cost: computes STL robustness metric (for each constraint) on the complete state trajectory history maintained by the simulator. This function is used by the simulator.py
# 2. stl_trajectory_cost: this function takes as input past STL robustness and a future predicted state and input trajectory to compute STL robustness (for all constraints combined) of concataneted past + future trajectory. This function is used by MPPI
### TODO: make this more efficient by passing the same function to simulator.py and MPPi codes


@jit
def stl_constraints(state_array, input_array):
    # low cost
    # but stl needs high reward
    """
    Returns a vector of constraint values g(x) such that the constraint is g(x)>=0
    Args:
        state: n X N
        input: m x N
    """
    # collision avoidance
    dists = jnp.linalg.norm(state_array[0:2, :] - obstacle.reshape(-1, 1), axis=0)
    h1 = 0.3 / jnp.maximum(
        (dists - obstacle_radius),
        0.01 + 0 * dists,
    )

    # Goal 1 reaching
    h2 = 10.0 * (
        jnp.linalg.norm(state_array[0:2, :] - goal.reshape(-1, 1), axis=0) ** 2 - goal_radius**2
    )

    # Goal 2 reaching
    h3 = 2.0 * (
        jnp.linalg.norm(state_array[0:2, :] - goal2.reshape(-1, 1), axis=0) ** 2 - goal_radius**2
    )

    h4 = 1.0 * (
        jnp.linalg.norm(state_array[0:2, :] - goal3.reshape(-1, 1), axis=0) ** 2 - goal_radius**2
    )

    return jnp.concatenate(
        (h1.reshape(1, -1), h2.reshape(1, -1), h3.reshape(1, -1), h4.reshape(1, -1)),
        axis=0,
    )


@jit
def stl_complete_trajectory_cost(dt: float, state_array: Array) -> Array:
    robustness = stl_constraints(state_array, None)
    time_stamps = jnp.linspace(0, state_array.shape[1] * dt, state_array.shape[1])
    h1 = jax_global(0, jnp.inf, -robustness[0, :], time_stamps)  # always satisfy this constraint
    h2 = jax_finally(
        0, 3.5, -robustness[1, :], time_stamps
    )  # reach this goal between [0, 3.5] seconds.
    h3 = jax_finally(
        3.6, 5, -robustness[2, :], time_stamps
    )  # reach this goal between [3.6, 5] seconds
    h4 = jax_finally(
        5.1, 10, -robustness[3, :], time_stamps
    )  # reach this goal between [5.1, 10] seconds
    return jnp.array([h1, h2, h3, h4])  # Note this function returns robustness of each constraint


# repeat the above function for MPPI but return a single robustness metric
@jit
def stl_trajectory_cost(
    time: float, state_array: Array, input_array: Array, prev_robustness
) -> Array:
    robustness = stl_constraints(state_array, input_array)

    time_stamps = jnp.linspace(
        time, time + mppi_args["prediction_horizon"] * DT, mppi_args["prediction_horizon"]
    )

    h1 = jax_global(0, jnp.inf, -robustness[0, :], time_stamps)
    h2 = jax_finally(0, 3.5, -robustness[1, :], time_stamps)
    h3 = jax_finally(3.6, 5, -robustness[2, :], time_stamps)
    h4 = jax_finally(5.1, 10, -robustness[3, :], time_stamps)

    h1 = jax_and(h1, prev_robustness[0])
    h2 = jax_or(h2, prev_robustness[1])
    h3 = jax_or(h3, prev_robustness[2])
    h4 = jax_or(h4, prev_robustness[3])

    cost = -jax_and(
        jax_and(
            h1,
            h2,
        ),
        # h3,
        jax_and(
            h3,
            h4,
        ),
    )

    return cost  # Note this function returns a single robustnees metric of all constraints


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


# Instantiate MPPI control law
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
mppi_local_planner = mppi_planner.vanilla_mppi(
    control_limits=ACTUATION_LIMITS,
    dynamics_func=dynamics,
    trajectory_cost=stl_trajectory_cost,
    stage_cost=None,  # ,stage_cost,
    terminal_cost=None,  # terminal_cost,
    mppi_args=mppi_args,
)

target_setpoint = single_waypoint_planner.vanilla_waypoint(target_state=goal)

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
    planner=mppi_local_planner,  # target_setpoint,  # ,  # ,  # None,  #
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
    stl_trajectory_cost=stl_complete_trajectory_cost,
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
