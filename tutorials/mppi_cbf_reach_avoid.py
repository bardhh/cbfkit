import os

import jax.numpy as jnp

# Import controllers and planners

# Simulation Parameters
file_path = os.path.dirname(os.path.abspath(__file__))
target_directory = file_path + "/generated"
model_name = "mppi_cbf_si"
SAVE_FILE = target_directory + f"/{model_name}/simulation_data"
DT = 0.05  # 1e-2
TF = 10 if not os.getenv("CBFKIT_TEST_MODE") else 0.5  # 10.0
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([0.0, 0.0])
ACTUATION_LIMITS = jnp.array([5.0, 5.0])  # Box control input constraint, i.e., -100 <= u <= 100

from typing import Any, Dict

# Initialize params dict for all dynamics and constraint function related parameters
params: Dict[str, Any] = {}

# Define  Dynamics
drift_dynamics = "[0, 0]"
control_matrix = "[[1, 0], [0, 1]]"

# Define functions
goal = jnp.array([9, 9])
obstacle = jnp.array([3, 3])
obstacle_radius = 0.6

# MPPI stage and terminal cost functions
stage_cost_function = (
    "0.2 * ( (x[0]-goal[0])**2 + (x[1]-goal[1])**2 )"
    "+ 10.0/max(array([(linalg.norm(x[0:2]-obstacle[0:2])-obstacle_radius),0.01]))"
)  # MPPI stage cost function
terminal_cost_function = (
    "0.2 * ( (x[0]-goal[0])**2 + (x[1]-goal[1])**2 ) "
    "+ 10.0/max(array([(linalg.norm(x[0:2]-obstacle[0:2])-obstacle_radius),0.01]))"
)  # MPPI terminal cost function

# Nominal controller - to be passed to CBF-QP controller
nominal_control_law = "-k_p * (x[0]-xd[0]), -k_p * (x[1]-xd[1])"  # nominal controller

# Barrier functions
state_constraint_funcs = ["(x[0]-obstacle[0])**2 + (x[1]-obstacle[1])**2 - obstacle_radius**2"]

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
