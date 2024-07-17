import os
from jax import Array, jit
import jax.numpy as jnp
from cbfkit.codegen.create_new_system import generate_model


# Simulation Parameters
file_path = os.path.dirname(os.path.abspath(__file__))
target_directory = file_path + "/tutorials"
model_name = "mppi_cbf_si"
SAVE_FILE = target_directory + f"/{model_name}/simulation_data"
DT = 0.05  # 1e-2
TF = 10  # 10.0
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([0.0, 0.0])
ACTUATION_LIMITS = jnp.array([5.0, 5.0])  # Box control input constraint, i.e., -100 <= u <= 100

# Initialize params dict for all dynamics and constraint function related parameters
params = {}

# Define  Dynamics
drift_dynamics = "[0, 0]"
control_matrix = "[[1, 0], [0, 1]]"

# Define functions
goal = jnp.array([9, 9])
obstacle = jnp.array([3, 3])
obstacle_radius = 0.6

# MPPI stage and terminal cost functions
stage_cost_function = "0.2 * ( (x[0]-goal[0])**2 + (x[1]-goal[1])**2 )+ 10.0/max(array([(linalg.norm(x[0:2]-obstacle[0:2])-obstacle_radius),0.01]))"  # MPPI stage cost function
terminal_cost_function = "0.2 * ( (x[0]-goal[0])**2 + (x[1]-goal[1])**2 ) + 10.0/max(array([(linalg.norm(x[0:2]-obstacle[0:2])-obstacle_radius),0.01]))"  # MPPI terminal cost function

# Nominal controller - to be passed to CBF-QP controller
nominal_control_law = "-k_p * (x[0]-xd[0]), -k_p * (x[1]-xd[1])"  # nominal controller

# Barrier functions
state_constraint_funcs = [
    "(x[0]-obstacle[0])**2 + (x[1]-obstacle[1])**2 - obstacle_radius**2"
] 

# Lyapunov functions
lyapunov_functions = ["(x[0]-goal[0])**2+(x[1]-goal[1])**2"]  # lyapunov functions
params["stage_cost_function"] = {
    "goal: float": goal,
    "obstacle: float": obstacle,
    "obstacle_radius: float": obstacle_radius,
}
params["terminal_cost_function"] = {
    "goal: float": goal,
    "obstacle: float": obstacle,
    "obstacle_radius: float": obstacle_radius,
}
params["controller"] = {"k_p: float": 1.0}
params["clf"] = [{"goal: float": goal}]
params["cbf"] = [{"obstacle: float": obstacle, "obstacle_radius: float": obstacle_radius}]

# Run script for automated genration of dynamics, cost functions python files
generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    stage_cost_function=stage_cost_function,
    terminal_cost_function=terminal_cost_function,
    barrier_funcs=state_constraint_funcs,
    lyapunov_funcs=lyapunov_functions,
    nominal_controller=nominal_control_law,
    params=params,
)

# Import controllers and planners
import cbfkit.simulation.simulator as sim
import cbfkit.controllers_and_planners.model_based.cbf_clf_controllers as cbf_clf_controllers
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
    e_s,
)
import cbfkit.controllers_and_planners.model_based.mppi as mppi_planner
import cbfkit.controllers_and_planners.waypoint as single_waypoint_planner
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator
from cbfkit.utils.numerical_integration import forward_euler as integrator
from tutorials import mppi_cbf_si

# Load dynamics, cost functions and constraint functions that were auto-generated
dynamics = mppi_cbf_si.plant()
stage_cost = mppi_cbf_si.cost_functions.stage_cost_function.stage_cost(
    goal=goal, obstacle=obstacle, obstacle_radius=obstacle_radius
)
terminal_cost = mppi_cbf_si.cost_functions.terminal_cost_function.terminal_cost(
    goal=goal, obstacle=obstacle, obstacle_radius=obstacle_radius
)
nominal_controller = mppi_cbf_si.controllers.controller_1(k_p=1.0)
b1 = mppi_cbf_si.certificate_functions.barrier_functions.cbf1_package(
    certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0),
    obstacle=obstacle,
    obstacle_radius=obstacle_radius,
)
barriers = concatenate_certificates(b1)
l1 = mppi_cbf_si.certificate_functions.lyapunov_functions.clf1_package(
    certificate_conditions=e_s(c=2.0), radius=1.0, goal=goal
)
lyapunov = concatenate_certificates(l1)

# MPPI specific parameters
mppi_args = {
    "robot_state_dim": 2,
    "robot_control_dim": 2,
    "prediction_horizon": 80,
    "num_samples": 20000,
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

# alternatively, instantiate a fixed waypoint planner
target_setpoint = single_waypoint_planner.vanilla_waypoint(target_state=goal)

# Instantiate CBF-CLF-QP control law
cbf_clf_controller = cbf_clf_controllers.vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    dynamics_func=dynamics,
    barriers=barriers,
    lyapunovs=lyapunov,
    relaxable_clf=True,
)

# Run the simulation
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
    planner=mppi_local_planner,  # target_setpoint, # None,
    nominal_controller=None,  # nominal_controller,
    controller=cbf_clf_controller,
    sensor=sensor,
    estimator=estimator,
    filepath=SAVE_FILE,
    planner_data={
        "u_traj": 3 * jnp.ones((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"]))
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
        desired_state=[goal],
        desired_state_radius=0.1,
        x_lim=(0, 10),
        y_lim=(0, 10),
        dt=DT,
        title="System Behavior",
        save_animation=True,
        animation_filename=target_directory + "/" + model_name + "/animation" + ".mp4",
        obstacle=obstacle,
        obstacle_radius=obstacle_radius,
    )
