import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.linalg import block_diag

import cbfkit.controllers.cbf_clf as cbf_clf_controllers
import cbfkit.controllers.mppi as mppi_planner
import cbfkit.simulation.simulator as sim

# Importing modules from cbfkit
from cbfkit.codegen.create_new_system import generate_model
from cbfkit.controllers.cbf_clf.utils.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf.utils.certificate_packager import concatenate_certificates
from cbfkit.controllers.cbf_clf.utils.lyapunov_conditions.exponential_stability import e_s
from cbfkit.controllers.cbf_clf.utils.rectify_relative_degree import rectify_relative_degree
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor

TARGET_DIRECTORY = "./tutorials"
MODEL_NAME = "multi_robot_3d_system"

# Simulation Parameters
NUM_ROBOTS = 4
DT = 0.05  # Time step
TF = 20  # Final time
N_STEPS = int(TF / DT) + 1
STATE_DIM_PER_ROBOT = 6  # Each robot has a 6-dimensional state
CONTROL_DIM_PER_ROBOT = 3  # Each robot has 3 control inputs
STATE_DIM = STATE_DIM_PER_ROBOT * NUM_ROBOTS
CONTROL_DIM = CONTROL_DIM_PER_ROBOT * NUM_ROBOTS
ACTUATION_LIMITS = jnp.full((CONTROL_DIM,), 10)

# Barrier Parameters
D_MIN_SQUARED = 0.1  # Minimum distance squared between robots
OBS_RADIUS = 8.0  # Obstacle sphere radius
OBS_RADIUS_SQUARED = OBS_RADIUS**2  # Obstacle radius squared for barrier

# Lyapunov Parameters
LYAPUNOV_C = 0.8  # Exponential stability constant

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

params = {
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
        # Position error weighted higher to ensure goal convergence
        cost += 20.0 * (
            (x[idx] - goals[idx]) ** 2
            + (x[idx + 1] - goals[idx + 1]) ** 2
            + (x[idx + 2] - goals[idx + 2]) ** 2
        ) + (x[idx + 3] ** 2 + x[idx + 4] ** 2 + x[idx + 5] ** 2)
    # Penalize control effort for smoother trajectories
    cost += 0.1 * jnp.sum(action**2)
    return cost


@jit
def terminal_cost(x: jnp.ndarray, action: jnp.ndarray) -> float:
    return 10.0 * stage_cost(x, action)


# Inter-robot collision avoidance: ||x_i - x_j||^2 - d_min^2 >= 0
state_constraint_funcs = [
    f"(x[{STATE_DIM_PER_ROBOT * i}] - x[{STATE_DIM_PER_ROBOT * j}])**2 + "
    f"(x[{STATE_DIM_PER_ROBOT * i + 1}] - x[{STATE_DIM_PER_ROBOT * j + 1}])**2 + "
    f"(x[{STATE_DIM_PER_ROBOT * i + 2}] - x[{STATE_DIM_PER_ROBOT * j + 2}])**2 - 0.25 "
    for i in range(NUM_ROBOTS)
    for j in range(i + 1, NUM_ROBOTS)
]

# Obstacle avoidance: ||x_i - obs_center||^2 - R^2 >= 0 (obstacle at origin)
state_constraint_funcs += [
    f"x[{STATE_DIM_PER_ROBOT * i}]**2 + "
    f"x[{STATE_DIM_PER_ROBOT * i + 1}]**2 + "
    f"x[{STATE_DIM_PER_ROBOT * i + 2}]**2 - {OBS_RADIUS_SQUARED} "
    for i in range(NUM_ROBOTS)
]


# lyapunov_functions = [
#     f"(x[{STATE_DIM_PER_ROBOT * i}] - goal[{STATE_DIM_PER_ROBOT * i}])**2 + "
#     f"(x[{STATE_DIM_PER_ROBOT * i + 1}] - goal[{STATE_DIM_PER_ROBOT * i + 1}])**2 + "
#     f"(x[{STATE_DIM_PER_ROBOT * i + 2}] - goal[{STATE_DIM_PER_ROBOT * i + 2}])**2"
#     for i in range(NUM_ROBOTS)
# ]


generate_model.generate_model(
    directory=TARGET_DIRECTORY,
    model_name=MODEL_NAME,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix_str,
    barrier_funcs=state_constraint_funcs,
    # lyapunov_funcs=lyapunov_functions,
    params=params,
)

import importlib

# Import the Generated Model
import tutorials.multi_robot_3d_system as multi_robot_3d_system
from cbfkit.utils.visualization import visualize_3d_multi_robot

# Create Barrier Functions with Linear Class K Function Derivative Conditions
rectified_barrier_packages = []
h = []
for i in range(len(state_constraint_funcs)):
    module_name = f"multi_robot_3d_system.certificate_functions.barrier_functions.barrier_{i+1}"
    module = importlib.import_module(module_name)
    h = getattr(module, "cbf")(alpha=5.0)
    # Apply rectify_relative_degree using the extracted function
    cbf_package = rectify_relative_degree(
        function=h,
        system_dynamics=multi_robot_3d_system.plant(),
        state_dim=STATE_DIM,
        form="exponential",
    )(certificate_conditions=zeroing_barriers.linear_class_k(alpha=10.0))

    # Store the rectified barrier package
    rectified_barrier_packages.append(cbf_package)


barriers = concatenate_certificates(*rectified_barrier_packages)

# Create Lyapunov Functions with Exponential Stability Derivative Conditions
# lyapunov = concatenate_certificates(
#     *[
#         getattr(
#             multi_robot_3d_system.certificate_functions.lyapunov_functions, f"clf{i + 1}_package"
#         )(
#             certificate_conditions=e_s(c=LYAPUNOV_C),
#             goal=goals,
#         )
#         for i in range(NUM_ROBOTS)
#     ]
# )

# Define MPPI Planner Arguments
mppi_args = {
    "robot_state_dim": STATE_DIM,
    "robot_control_dim": CONTROL_DIM,
    "prediction_horizon": 120,
    "num_samples": 12500,
    "plot_samples": 0,
    "time_step": DT,
    "use_GPU": False,
    "costs_lambda": 0.2,
    "cost_perturbation": 1.5,
}

# Instantiate MPPI Control Law
mppi_local_planner = mppi_planner.vanilla_mppi(
    control_limits=ACTUATION_LIMITS,
    dynamics_func=multi_robot_3d_system.plant(),
    trajectory_cost=None,
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    mppi_args=mppi_args,
)


# Instantiate CBF-CLF-QP Control Law
cbf_clf_controller = cbf_clf_controllers.vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    dynamics_func=multi_robot_3d_system.plant(),
    barriers=barriers,
    # lyapunovs=lyapunov,
    relaxable_clf=True,
)

# ================================
# Run the Simulation
# ================================

simulation_data_path = os.path.join(TARGET_DIRECTORY, MODEL_NAME, "simulation_data")

results = sim.execute(
    x0=INITIAL_STATE,
    dt=DT,
    num_steps=N_STEPS,
    dynamics=multi_robot_3d_system.plant(),
    integrator=integrator,
    planner=mppi_local_planner,
    # nominal_controller=nominal_controller,
    controller=cbf_clf_controller,
    sensor=sensor,
    estimator=estimator,
    filepath=simulation_data_path,
    planner_data={
        "u_traj": 3 * jnp.ones((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"]))
    },
    controller_data={},
)

# ================================
# Animate the Results
# ================================

# Obstacle at the origin (ellipsoid with identity rotation)
obstacle_centers = [np.array([0.0, 0.0, 0.0])]
obstacle_radii = [np.array([OBS_RADIUS, OBS_RADIUS, OBS_RADIUS])]
obstacle_rotations = [np.eye(3)]

animation_path = os.path.abspath(
    os.path.join(TARGET_DIRECTORY, MODEL_NAME, f"animation_{NUM_ROBOTS}_robots.gif")
)

visualize_3d_multi_robot(
    states=results.states,
    desired_states=goals,
    desired_state_radius=0.3,
    num_robots=NUM_ROBOTS,
    state_dimension_per_robot=STATE_DIM_PER_ROBOT,
    dt=DT,
    title="Multi-Robot Trajectory",
    save_animation=True,
    animation_filename=animation_path,
    include_min_distance_plot=True,
    ellipse_centers=obstacle_centers,
    ellipse_radii=obstacle_radii,
    ellipse_rotations=obstacle_rotations,
    backend="manim-low",  # Options: manim-low, manim-medium, manim-high, manim-production
)

print(f"\nAnimation saved to: file://{animation_path}")
