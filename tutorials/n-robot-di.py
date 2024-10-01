import os
import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy.linalg import block_diag

from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.rectify_relative_degree import (
    rectify_relative_degree,
)
from functools import partial

# Importing modules from cbfkit
from cbfkit.codegen.create_new_system import generate_model
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

from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.risk_aware_params import (
    RiskAwareParams,
)
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator
from cbfkit.utils.numerical_integration import forward_euler as integrator
import cbfkit.controllers_and_planners.model_based.mppi as mppi_planner


TARGET_DIRECTORY = "./tutorials"
MODEL_NAME = "nicole_sys"

# Simulation Parameters
NUM_ROBOTS = 2
DT = 0.05  # Time step
TF = 10  # Final time
N_STEPS = int(TF / DT) + 1
STATE_DIM_PER_ROBOT = 6  # Each robot has a 6-dimensional state
CONTROL_DIM_PER_ROBOT = 3  # Each robot has 3 control inputs
STATE_DIM = STATE_DIM_PER_ROBOT * NUM_ROBOTS
CONTROL_DIM = CONTROL_DIM_PER_ROBOT * NUM_ROBOTS
ACTUATION_LIMITS = jnp.full((CONTROL_DIM,), 10)
# ACTUATION_LIMITS = jnp.array([100.0])
# ACTUATION_LIMITS = ACTUATION_LIMITS.reshape(6, 2)
# ACTUATION_LIMITS = jnp.tile(jnp.array([100.0, 100.0]), (6, 2))

# Controller Parameters
KP = 0.1  # Proportional gain
KD = 0.1  # Derivative gain

# Barrier Parameters
D_MIN_SQUARED = 0.1  # Minimum distance squared between robots

# Lyapunov Parameters
LYAPUNOV_C = 0.5  # Exponential stability constant

# ================================
# Initialize States and Goals
# ================================

INITIAL_STATE = np.zeros(STATE_DIM)
goals = np.zeros(STATE_DIM)

radius = 15
for i in range(NUM_ROBOTS):
    angle = 2 * np.pi * i / NUM_ROBOTS
    idx = STATE_DIM_PER_ROBOT * i

    # Initial positions (distributed in a circle with some noise)
    INITIAL_STATE[idx] = radius * np.cos(angle) + np.random.normal(0.5, 1)  # x
    INITIAL_STATE[idx + 1] = radius * np.sin(angle) + np.random.normal(0.5, 1)  # y
    INITIAL_STATE[idx + 2] = np.random.normal(-3, 3)  # z

    # Initial velocities
    INITIAL_STATE[idx + 3 : idx + 6] = np.random.normal(0.05, 0.1)  # vx, vy, vz

    # Goals (opposite side of the circle)
    goals[idx : idx + 3] = -INITIAL_STATE[idx : idx + 3]  # x_goal, y_goal, z_goal
    goals[idx + 3 : idx + 6] = 0  # vx_goal, vy_goal, vz_goal

# # Define Specific Goals for Each Robot as a List of Tuples
# specific_goals = [
#     (5, 0.0, 0.0),  # Robot 1
#     (0.0, 10.0, 0.0),  # Robot 2
#     (-5.0, 0.0, 0.0),  # Robot 3
#     (0.0, -5, 0.0),  # Robot 4
#     (0.0, 0.0, 8.0),  # Robot 5
# ]

# for i, goal_pos in enumerate(specific_goals):
#     idx = STATE_DIM_PER_ROBOT * i
#     goals[idx : idx + 3] = goal_pos  # x_goal, y_goal, z_goal
#     goals[idx + 3 : idx + 6] = 0.0  # vx_goal, vy_goal, vz_goal

# ================================
# Define Dynamics and Control Matrix
# ================================

# Drift Dynamics: dx/dt = velocity, dv/dt = 0
drift_dynamics = []
for i in range(NUM_ROBOTS):
    idx = STATE_DIM_PER_ROBOT * i
    drift_dynamics.extend(
        [
            f"x[{idx + 3}]",  # dx/dt = vx
            f"x[{idx + 4}]",  # dy/dt = vy
            f"x[{idx + 5}]",  # dz/dt = vz
            "0",  # dvx/dt = 0
            "0",  # dvy/dt = 0
            "0",  # dvz/dt = 0
        ]
    )

# Control Matrix: Controls affect accelerations
control_matrix = block_diag(
    *[
        np.array(
            [
                [0, 0, 0],  # No control over positions
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],  # Control inputs affect accelerations
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        for _ in range(NUM_ROBOTS)
    ]
)
control_matrix_str = np.array2string(control_matrix, separator=",", threshold=np.inf).replace(
    "\n", ""
)

# ================================
# Define Nominal Control Law (PD Controller)
# ================================

nominal_control_law = "["
nominal_control_law += ",".join(
    [
        f"-k_p * (x[{STATE_DIM_PER_ROBOT * i}] - goal[{STATE_DIM_PER_ROBOT * i}]),"
        f"-k_p * (x[{STATE_DIM_PER_ROBOT * i + 1}] - goal[{STATE_DIM_PER_ROBOT * i + 1}]),"
        f"-k_p * (x[{STATE_DIM_PER_ROBOT * i + 2}] - goal[{STATE_DIM_PER_ROBOT * i + 2}])"
        for i in range(NUM_ROBOTS)
    ]
)
nominal_control_law += "]"

control_terms = []

# POSITION_THRESHOLD = 0.2

# for i in range(NUM_ROBOTS):
#     idx = STATE_DIM_PER_ROBOT * i
#     # PD control for x, y, z
#     control_x = f"-k_p * (x[{idx}] - goal[{idx}]) - k_d * x[{idx + 3}]"
#     control_y = f"-k_p * (x[{idx + 1}] - goal[{idx + 1}]) - k_d * x[{idx + 4}]"
#     control_z = f"-k_p * (x[{idx + 2}] - goal[{idx + 2}]) - k_d * x[{idx + 5}]"

#     # Implement threshold using jnp.where
#     condition = f"((x[{idx}] - goal[{idx}])**2 + (x[{idx + 1}] - goal[{idx + 1}])**2 + (x[{idx + 2}] - goal[{idx + 2}])**2) > {POSITION_THRESHOLD}**2"
#     control_x = f"jnp.where({condition}, {control_x}, 0)"
#     control_y = f"jnp.where({condition}, {control_y}, 0)"
#     control_z = f"jnp.where({condition}, {control_z}, 0)"

#     control_terms.extend([control_x, control_y, control_z])

# nominal_control_law += ",".join(control_terms)
# nominal_control_law += "]"  # End of the control input list


params = {
    "controller": {
        "goal: float": goals,
        "k_p: float": KP,
        "k_d: float": KD,
    },
    "barrier": {
        "D_MIN_SQUARED: float": D_MIN_SQUARED,
    },
    "clf": [{"goal: float": goals} for _ in range(NUM_ROBOTS)],
}


@jit
def stage_cost(x: jnp.ndarray, action: jnp.ndarray) -> float:
    cost = 0.0
    for i in range(NUM_ROBOTS):
        idx = STATE_DIM_PER_ROBOT * i
        # Sum the squared distances for each robot (positions only)
        cost += (
            (x[idx] - goals[idx]) ** 2
            + (x[idx + 1] - goals[idx + 1]) ** 2
            + (x[idx + 2] - goals[idx + 2]) ** 2
            + (x[idx + 3] ** 2 + x[idx + 4] ** 2 + x[idx + 5] ** 2)
        )
    return cost


@jit
def terminal_cost(x: jnp.ndarray, action: jnp.ndarray) -> float:
    # Assuming terminal cost is the same as stage cost
    return 10.0 * stage_cost(x, action)


state_constraint_funcs = [
    f"(x[{STATE_DIM_PER_ROBOT * i}] - x[{STATE_DIM_PER_ROBOT * j}])**2 + "
    f"(x[{STATE_DIM_PER_ROBOT * i + 1}] - x[{STATE_DIM_PER_ROBOT * j + 1}])**2 + "
    f"(x[{STATE_DIM_PER_ROBOT * i + 2}] - x[{STATE_DIM_PER_ROBOT * j + 2}])**2 - 0.25 "
    for i in range(NUM_ROBOTS)
    for j in range(i + 1, NUM_ROBOTS)
]


lyapunov_functions = [
    f"(x[{STATE_DIM_PER_ROBOT * i}] - goal[{STATE_DIM_PER_ROBOT * i}])**2 + "
    f"(x[{STATE_DIM_PER_ROBOT * i + 1}] - goal[{STATE_DIM_PER_ROBOT * i + 1}])**2 + "
    f"(x[{STATE_DIM_PER_ROBOT * i + 2}] - goal[{STATE_DIM_PER_ROBOT * i + 2}])**2"
    for i in range(NUM_ROBOTS)
]


generate_model.generate_model(
    directory=TARGET_DIRECTORY,
    model_name=MODEL_NAME,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix_str,
    barrier_funcs=state_constraint_funcs,
    lyapunov_funcs=lyapunov_functions,
    nominal_controller=nominal_control_law,
    params=params,
)

# from tutorials import nicole_sys
from tutorials.plot_helper.plot_3d_multi_robot import animate_3d_multi_robot

# Import the Generated Model
import tutorials.nicole_sys as nicole_sys  # Adjusted import for clarity
import importlib

# Create Barrier Functions with Linear Class K Function Derivative Conditions
rectified_barrier_packages = []
h = []
for i in range(len(state_constraint_funcs)):
    # Get the barrier function package
    # h_package = getattr(nicole_sys.certificate_functions.barrier_functions, f"cbf{i + 1}_package")(
    #     certificate_conditions=zeroing_barriers.linear_class_k(alpha=5.0)
    # )

    # Extract the h_ function from the package
    # h_function = h_package[2][0]  # h_ is at index 2 and is inside a list
    # from nicole_sys.certificate_functions.barrier_functions.cbf1_package import cbf
    # h = nicole_sys.certificate_functions.barrier_functions.barrier_1.cbf
    module_name = f"nicole_sys.certificate_functions.barrier_functions.barrier_{i+1}"
    module = importlib.import_module(module_name)
    h = getattr(module, "cbf")(alpha=5.0)
    # Apply rectify_relative_degree using the extracted function
    cbf_package = rectify_relative_degree(
        function=h,
        system_dynamics=nicole_sys.plant(),
        state_dim=STATE_DIM,
        form="exponential",
    )(certificate_conditions=zeroing_barriers.linear_class_k(alpha=5.0))

    # Store the rectified barrier package
    rectified_barrier_packages.append(cbf_package)


# cbf_package = []

# for i in range(len(state_constraint_funcs)):
#     h = getattr(nicole_sys.certificate_functions.barrier_functions, f"cbf{i + 1}_package")(
#         certificate_conditions=zeroing_barriers.linear_class_k(alpha=5.0)
#     )
#     cbf_package.append(
#         rectify_relative_degree(
#             function=h,
#             system_dynamics=nicole_sys.plant(),
#             state_dim=6,
#             form="exponential",
#         )
#     )


# # # Create Barrier Functions with Linear Class K Function Derivative Conditions
# barriers = concatenate_certificates(
#     *[
#         getattr(nicole_sys.certificate_functions.barrier_functions, f"cbf{i + 1}_package")(
#             certificate_conditions=zeroing_barriers.linear_class_k(alpha=5.0),
#             d_min_squared=D_MIN_SQUARED,  # (0.5)^2
#         )
#         for i in range(len(state_constraint_funcs))
#     ]
# )


# Importing modules from tutorials

from cbfkit.systems.fixed_wing_uav.models.beard2014_kinematic.certificate_functions.barrier_functions.obstacle_avoidance.high_order import (
    cbf,
)


# Instantiate the Nominal Controller
nominal_controller = nicole_sys.controllers.controller_1(goal=goals, k_p=KP, k_d=KD)

# # Create Barrier Functions with Linear Class K Function Derivative Conditions
# rectified_barrier_packages = []
# for i in range(len(state_constraint_funcs)):
#     # Get the original barrier function
#     barrier_func = getattr(
#         nicole_sys.certificate_functions.barrier_functions, f"cbf{i + 1}_package"
#     )

#     # Create a partial function with the required parameters
#     barrier_func_with_params = partial(barrier_func, d_min_squared=D_MIN_SQUARED)

#     # Apply rectify_relative_degree
#     cbf_package = rectify_relative_degree(
#         function=barrier_func_with_params,
#         system_dynamics=nicole_sys.plant(),
#         state_dim=STATE_DIM,
#         form="exponential",
#     )

#     # Store the rectified barrier package
#     rectified_barrier_packages.append(cbf_package)

# # Create Barrier Functions with Linear Class K Function Derivative Conditions

# barriers = concatenate_certificates(
#     *[
#         rectified_barrier_packages[i](
#             certificate_conditions=zeroing_barriers.linear_class_k(alpha=5.0),
#         )
#         for i in range(len(rectified_barrier_packages))
#     ]
# )

barriers = concatenate_certificates(*rectified_barrier_packages)

# barriers = rectified_barrier_packages

# Create Lyapunov Functions with Exponential Stability Derivative Conditions
lyapunov = concatenate_certificates(
    *[
        getattr(nicole_sys.certificate_functions.lyapunov_functions, f"clf{i + 1}_package")(
            certificate_conditions=e_s(c=LYAPUNOV_C),
            goal=goals,
        )
        for i in range(NUM_ROBOTS)
    ]
)

# Define MPPI Planner Arguments
mppi_args = {
    "robot_state_dim": STATE_DIM,
    "robot_control_dim": CONTROL_DIM,
    "prediction_horizon": 120,
    "num_samples": 600,
    "plot_samples": 30,
    "time_step": DT,
    "use_GPU": True,
    "costs_lambda": 0.05,
    "cost_perturbation": 0.2,
}

# Instantiate MPPI Control Law
mppi_local_planner = mppi_planner.vanilla_mppi(
    control_limits=ACTUATION_LIMITS,
    dynamics_func=nicole_sys.plant(),
    trajectory_cost=None,
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    mppi_args=mppi_args,
)


# Instantiate CBF-CLF-QP Control Law
# cbf_clf_controller = cbf_clf_controllers.vanilla_cbf_clf_qp_controller(
cbf_clf_controller = cbf_clf_controllers.vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    nominal_input=nominal_controller,
    dynamics_func=nicole_sys.plant(),
    barriers=barriers,
    lyapunovs=lyapunov,
    # ra_clf_params=ra_clf_params,
    relaxable_clf=True,
    # disturbance_norm=2,
    # disturbance_norm_bound=2.0,
)

# ================================
# Run the Simulation
# ================================

simulation_data_path = os.path.join(TARGET_DIRECTORY, MODEL_NAME, "simulation_data")

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
    dynamics=nicole_sys.plant(),
    integrator=integrator,
    planner=mppi_local_planner,
    nominal_controller=nominal_controller,
    controller=cbf_clf_controller,
    sensor=sensor,
    estimator=estimator,
    filepath=simulation_data_path,  # Automatically uses .csv format
    planner_data={
        "u_traj": 3 * jnp.ones((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"]))
    },
    # planner_data={
    #     "x_traj": jnp.zeros((2, 1)),
    # },  # pass in a dummy state since we need to pass planner_data
    controller_data={},
)

# ================================
# Animate the Results
# ================================

animation_path = os.path.join(TARGET_DIRECTORY, MODEL_NAME, f"animation_{NUM_ROBOTS}_robots.gif")

animate_3d_multi_robot(
    states=x_,
    desired_states=goals,
    desired_state_radius=0.3,
    num_robots=NUM_ROBOTS,
    state_dimension_per_robot=STATE_DIM_PER_ROBOT,
    x_lim=(-15, 15),
    y_lim=(-15, 15),
    z_lim=(-10, 10),
    dt=DT,
    title="Multi-Robot Trajectory",
    save_animation=True,
    animation_filename=animation_path,
    include_min_distance_plot=True,
)
