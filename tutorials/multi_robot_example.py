'''
This file simulates a centralized multi robot system. Decentralzied implementation will be supported in future releases of this library.
'''
import os
from jax import Array, jit
import jax.numpy as jnp
import numpy as np

from cbfkit.codegen.create_new_system import generate_model

file_path = os.path.dirname(os.path.abspath(__file__))
target_directory = file_path + "/tutorials"
model_name = "multi_augmented_single_integrators"

# Simulation Parameters
num_robots = 10
DT = 0.05 
TF = 10 
N_STEPS = int(TF / DT) + 1
ACTUATION_LIMITS = 100 * jnp.ones(2 * num_robots)

INITIAL_STATE = np.zeros(2 * num_robots)
goals = np.zeros(2 * num_robots)
for i in range(num_robots):
    theta_disturbance = np.clip(np.random.normal(0, 1.0), -np.pi / 60, np.pi / 60)
    INITIAL_STATE[2 * i] = 4 * np.cos(2 * np.pi * i / num_robots + theta_disturbance)
    INITIAL_STATE[2 * i + 1] = 4 * np.sin(2 * np.pi * i / num_robots + theta_disturbance)
    goals[2 * i] = -4 * np.cos(2 * np.pi * i / num_robots) + 0.1
    goals[2 * i + 1] = -4 * np.sin(2 * np.pi * i / num_robots) + 0.1

# Some example initial state and goals
# INITIAL_STATE = jnp.array([-0.9, -0.9, 0.9, 0.9])
# goals = jnp.array([1.0, 2.0, -0.6, -0.6])

# INITIAL_STATE = jnp.array([-0.9, -0.9, 0.9, 0.9])
# goals = jnp.array([1.0, 1.0, -1.0, -1.0])

# Dynamics
params = {}  
drift_dynamics_single_robot = np.array([0, 0])
control_matrix_single_robot = np.eye(2)  #  "[[1, 0], [0, 1]]"
drift_dynamics = np.tile(drift_dynamics_single_robot, num_robots)
control_matrix = np.kron(np.eye(num_robots), control_matrix_single_robot)
drift_dynamics = np.array2string(drift_dynamics, separator=",").replace("\n", "")
control_matrix = np.array2string(control_matrix, separator=",").replace("\n", "")

# Nominal Control Law
nominal_control_law = "["
for i in range(num_robots):
    if i < (num_robots - 1):
        nominal_control_law = (
            nominal_control_law
            + f" -k_p * (x[{2*i}]-goal[{2*i}]), -k_p * (x[{2*i+1}]-goal[{2*i+1}]),"
        )
    else:
        nominal_control_law = (
            nominal_control_law
            + f" -k_p * (x[{2*i}]-goal[{2*i}]), -k_p * (x[{2*i+1}]-goal[{2*i+1}])"
        )
nominal_control_law = nominal_control_law + "]"
params["controller"] = {
    "goal: float": goals,
    "k_p: float": 1.0,
}

# State Constraints
state_constraint_funcs = []
for i in range(num_robots):
    for j in range(i + 1, num_robots):
        state_constraint_funcs.append(f"(x[{2*i}]-x[{2*j}])**2 + (x[{2*i+1}]-x[{2*j+1}])**2 - 0.1")

params["clf"] = []
lyapunov_functions = []
for i in range(num_robots):
    lyapunov_functions.append(f"(x[{2*i}]-goal[0])**2+(x[{2*i+1}]-goal[1])**2")
    params["clf"].append(
        {
            "goal: float": goals[2 * i : 2 * i + 2],
        }
    )

# Run script for automated genration of dynamics, cost functions python files
generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    barrier_funcs=state_constraint_funcs,
    lyapunov_funcs=lyapunov_functions,
    nominal_controller=nominal_control_law,
    params=params,
)

# Provides access to execute (sim.execute)
import cbfkit.simulation.simulator as sim

# Access to CBF-CLF-QP control law
import cbfkit.controllers_and_planners.model_based.cbf_clf_controllers as cbf_clf_controllers

# Necessary housekeeping for using multiple CBFs/CLFs
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)

# Suite of zeroing barrier function derivative conditions (forms of Class K functions)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)

# Exponentially stable derivative condition for CLF
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
    e_s,
)

# Assuming perfect, complete state information
from cbfkit.sensors import perfect as sensor

# With perfect sensing, we can use a naive estimate of the state
from cbfkit.estimators import naive as estimator

# Use forward-Euler numerical integration scheme
from cbfkit.utils.numerical_integration import forward_euler as integrator

from tutorials import multi_augmented_single_integrators

# Simulation Parameters
SAVE_FILE = f"tutorials/{model_name}/simulation_data"  # automatically uses .csv format


# Dynamics function with epsilon parameter: returns f(x), g(x), d(x)
dynamics = multi_augmented_single_integrators.plant()

# Create barrier functions with linear class K function derivative conditions
bs = []
for i in range(len(state_constraint_funcs)):
    func_name = (
        "multi_augmented_single_integrators.certificate_functions.barrier_functions."
        + "cbf"
        + str(i + 1)
        + "_package"
    )
    exec(f"func={func_name}")
    # func = getattr(__name__, func_name)
    bs.append(func(certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0)))
barriers = concatenate_certificates(*bs)  # , b2)

# Create lyapunov function with exponential stability derivative condition
ls = []
for i in range(len(lyapunov_functions)):
    func_name = (
        "multi_augmented_single_integrators.certificate_functions.lyapunov_functions."
        + "clf"
        + str(i + 1)
        + "_package"
    )
    exec(f"func={func_name}")
    ls.append(
        func(
            certificate_conditions=e_s(c=2.0),
            goal=goals[2 * i : 2 * i + 2],
        )
    )
lyapunov = concatenate_certificates(*ls)

# Instantiate nominal controller
nominal_controller = multi_augmented_single_integrators.controllers.controller_1(goal=goals, k_p=1)

# Instantiate CBF-CLF-QP control law
cbf_clf_controller = cbf_clf_controllers.vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    nominal_input=nominal_controller,
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
    planner=None,
    nominal_controller=nominal_controller,
    controller=cbf_clf_controller,
    sensor=sensor,
    estimator=estimator,
    filepath=SAVE_FILE,
    planner_data={
        "x_traj": jnp.zeros((2, 1)),
    },  # pass in a dummy state since we need to pass planner_data
    controller_data={},
)

plot = True
if plot:
    from tutorials.plot_helper.plot_multi_single_integrators import animate

    animate(
        states=x_,
        estimates=z_,
        desired_state=goals,
        desired_state_radius=0.1,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        dt=DT,
        title="System Behavior",
        save_animation=True,
        animation_filename=target_directory + "/" + model_name + "/animation" + ".gif",
        num_robots=num_robots,
    )
