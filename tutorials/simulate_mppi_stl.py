import os
from jax import Array, jit
import jax.numpy as jnp
from cbfkit.codegen.create_new_system import generate_model
from typing import List, Callable

file_path = os.path.dirname(os.path.abspath(__file__))
target_directory = file_path + "/tutorials"
model_name = "mppi_cbf_stl_v2"

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
# obstacle_radius = 1.0  # 0.6

# Generate files automatically
params = {}
drift_dynamics = "[0, 0]"  # Dynamics
control_matrix = "[[1, 0], [0, 1]]"  # Dynamics
goal = jnp.array([4, 4])
goal2 = jnp.array([6, 9])
goal3 = jnp.array([4, 1])
# goal4 = jnp.array([2, 1])
goals = [goal, goal2, goal3]  # , goal3]
obstacle = jnp.array([3, 3])
obstacle_radius = 0.6  # 1.0  # 0.6
goal_threshold = 0.5
# stage_cost_function = "(x[0]-goal[0])**2 + (x[1]-goal[1])**2 + 10.0/(linalg.norm(x[0:2]-obstacle[0:2])-obstacle_radius)"  # MPPI stage cost function
# terminal_cost_function = "(x[0]-goal[0])**2 + (x[1]-goal[1])**2 + 10.0/max(array([(linalg.norm(x[0:2]-obstacle[0:2])-obstacle_radius),0.01]))"  # MPPI terminal cost function

# params["stage_cost_function"] = {
#     "goal: float": goal,
#     "obstacle: float": obstacle,
#     "obstacle_radius: float": obstacle_radius,
# }
# params["terminal_cost_function"] = {
#     "goal: float": goal,
#     "obstacle: float": obstacle,
#     "obstacle_radius: float": obstacle_radius,
# }

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


from jax_stl import *

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


# Example 1 (without STL)
# def constraints(state_array, input_array):
#     """
#     Returns a vector of constraint values g(x) such that the constraint is g(x)>=0
#     Args:
#         state: n X N
#         input: m x N
#     """
#     h1 = 0.2 * jnp.sum(jnp.linalg.norm(state_array[0:2, :] - goal.reshape(-1, 1), axis=0) ** 2)
#     h2 = jnp.sum(
#         10.0
#         / jnp.maximum(
#             (
#                 jnp.linalg.norm(state_array[0:2, :] - obstacle.reshape(-1, 1), axis=0)
#                 - obstacle_radius
#             ),
#             0.01 * jnp.ones(mppi_args["prediction_horizon"]),
#         )
#     )
#     return jnp.append(h1.reshape(1, -1), h2.reshape(1, -1), axis=0)


# @jit
# def trajectory_cost(state_array: Array, input_array: Array) -> Array:
#     robustness = constraints(state_array, input_array)
#     return jnp.sum(robustness)


# Example 2 (with STL)
@jit
def constraints(state_array, input_array):
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
    # h1 = 1.0 / jnp.maximum(
    #     (jnp.linalg.norm(state_array[0:2, :] - obstacle.reshape(-1, 1), axis=0) - obstacle_radius),
    #     0.01 * jnp.ones(mppi_args["prediction_horizon"]),
    # )
    # h1 = 10.0 * (
    #     obstacle_radius**2
    #     - jnp.linalg.norm(state_array[0:2, :] - obstacle.reshape(-1, 1), axis=0) ** 2
    # )
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

    # h5 = 1.0 * (
    #     jnp.linalg.norm(state_array[0:2, :] - goal4.reshape(-1, 1), axis=0) ** 2 - goal_radius**2
    # )

    # min value 0, max value infinity. want to minimze cost (that is, should be 0)
    # return jnp.append(h1.reshape(1, -1), h2.reshape(1, -1), axis=0)
    return jnp.concatenate(
        # (h1.reshape(1, -1), h2.reshape(1, -1), h3.reshape(1, -1)),
        # axis=0,
        (h1.reshape(1, -1), h2.reshape(1, -1), h3.reshape(1, -1), h4.reshape(1, -1)),
        axis=0,
    )


# @jit
# def trajectory_cost(time: float, state_array: Array, input_array: Array) -> Array:
#     robustness = constraints(state_array, input_array)
#     # time = state_array[-1]
#     time_stamps = jnp.linspace(
#         time, time + mppi_args["prediction_horizon"] * DT, mppi_args["prediction_horizon"]
#     )
#     cost = -jax_and(
#         jax_and(
#             jax_global(
#                 0,
#                 jnp.inf,
#                 -robustness[0, :],
#                 time_stamps,
#             ),  # jnp.linspace(0, mppi_args["prediction_horizon"] * DT, mppi_args["prediction_horizon"]),
#             jax_finally(
#                 0,
#                 5,
#                 -robustness[1, :],
#                 time_stamps,
#             ),  # mejnp.linspace(0, mppi_args["prediction_horizon"] * DT, mppi_args["prediction_horizon"]),
#         ),
#         jax_finally(5, 10, -robustness[2, :], time_stamps),
#         # jax_and(
#         #     jax_finally(5, 10, -robustness[2, :], time_stamps),
#         #     jax_finally(8, 20, -robustness[3, :], time_stamps),
#         # ),
#     )

#     return cost


@jit
def trajectory_cost_bad(
    time: float, state_array: Array, input_array: Array, prev_robustness
) -> Array:
    robustness = constraints(state_array, input_array)
    # time = state_array[-1]
    time_stamps = jnp.linspace(
        time, time + mppi_args["prediction_horizon"] * DT, mppi_args["prediction_horizon"]
    )
    # jax.debug.print("🤯 {x} 🤯", x=prev_robustness)
    # h1 = jax_global(0, jnp.inf, -robustness[0, :], time_stamps)
    # h2 = jax_finally(0, 5, -robustness[1, :], time_stamps)
    # h3 = jax_finally(5, 10, -robustness[2, :], time_stamps)

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
            h1,  # jnp.linspace(0, mppi_args["prediction_horizon"] * DT, mppi_args["prediction_horizon"]),
            h2,  # mejnp.linspace(0, mppi_args["prediction_horizon"] * DT, mppi_args["prediction_horizon"]),
        ),
        # h3,
        jax_and(
            h3,
            h4,
        ),
    )

    return cost  # , jnp.array([h1, h2, h3])

    # return jnp.max(jnp.array([jnp.max(robustness[0, :]), jnp.max(robustness[1, :])]))


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
    trajectory_cost=trajectory_cost_bad,
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
)

plot = True
if plot:
    from tutorials.plot_mppi_ffmpeg import animate

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
