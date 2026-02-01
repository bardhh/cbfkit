import os
import sys
from typing import List, Tuple, TypedDict

import jax.numpy as jnp
from jax import Array, jit

# Add the project root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import cbfkit.controllers.mppi as mppi_planner
import cbfkit.simulation.simulator as sim
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems.unicycle.models.accel_unicycle import plant
from cbfkit.utils.user_types import PlannerData
from examples.unicycle.common.ellipsoidal_obstacle import cbf as ellipsoid_cbf


# Define a TypedDict for MPPI configuration to ensure type safety and clarity
class MPPIConfig(TypedDict):
    robot_state_dim: int
    robot_control_dim: int
    prediction_horizon: int
    num_samples: int
    plot_samples: int
    time_step: float
    use_GPU: bool
    costs_lambda: float
    cost_perturbation: float


# Simulation parameters
tf: float = 10.0
dt: float = 0.05
file_path: str = "examples/unicycle/reach_goal/results/"

# Define states and constraints
init_state: Array = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
desired_state: Array = jnp.array([2.0, 4.0, 0.0, 0.0])
actuation_constraints: Array = jnp.array([100.0, 100.0])  # Effectively, no control limits

# Define system dynamics
unicycle_dynamics = plant(lam=1.0)


# MPPI Cost Functions
# We use @jit here to compile these functions with JAX.
# This is crucial for MPPI performance as these cost functions are evaluated
# thousands of times per step (num_samples * prediction_horizon).
@jit
def stage_cost(state_and_time: Array, action: Array) -> Array:
    """Calculates the stage cost for the MPPI planner.

    Args:
        state_and_time (Array): The current state and time [x, y, v, theta, t]
        action (Array): The control action [a, omega]

    Returns
    -------
        Array: The scalar cost.
    """
    # state: [x, y, v, theta]
    # action: [a, omega]
    x, y = state_and_time[0], state_and_time[1]
    xd, yd = desired_state[0], desired_state[1]

    dist_sq = (x - xd) ** 2 + (y - yd) ** 2
    return 10.0 * dist_sq


@jit
def terminal_cost(state_and_time: Array, action: Array) -> Array:
    """Calculates the terminal cost for the MPPI planner.

    Args:
        state_and_time (Array): The current state and time [x, y, v, theta, t]
        action (Array): The control action [a, omega]

    Returns
    -------
        Array: The scalar cost.
    """
    x, y = state_and_time[0], state_and_time[1]
    xd, yd = desired_state[0], desired_state[1]

    dist_sq = (x - xd) ** 2 + (y - yd) ** 2
    return 100.0 * dist_sq


# MPPI Configuration
# This dictionary controls the behavior of the Model Predictive Path Integral controller.
mppi_args: MPPIConfig = {
    "robot_state_dim": 4,
    "robot_control_dim": 2,
    "prediction_horizon": 50,
    "num_samples": 500,
    "plot_samples": 30,
    "time_step": dt,
    "use_GPU": True,
    "costs_lambda": 0.1,
    "cost_perturbation": 0.5,
}

# Instantiate MPPI Planner (as nominal controller replacement)
mppi_local_planner = mppi_planner.vanilla_mppi(
    control_limits=actuation_constraints,
    dynamics_func=unicycle_dynamics,
    trajectory_cost=None,
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    mppi_args=mppi_args,
)

# Define Obstacles
obstacles: List[Tuple[float, float, float]] = [
    (1, 2.0, 0.0),
    (3.0, 2.0, 0.0),
    (2.0, 5.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.5, -1.0, 0.0),
]
ellipsoids: List[Tuple[float, float]] = [
    (0.5, 1.5),
    (0.75, 2.0),
    (2.0, 0.25),
    (1.0, 0.75),
    (0.75, 0.5),
]

# Construct Control Barrier Functions (CBFs)
barriers = [
    rectify_relative_degree(
        function=ellipsoid_cbf(obs, ell),
        system_dynamics=unicycle_dynamics,
        state_dim=len(init_state),
        form="exponential",
    )(
        certificate_conditions=zeroing_barriers.linear_class_k(5.0),
        obstacle=obs,
        ellipsoid=ell,
    )
    for obs, ell in zip(obstacles, ellipsoids)
]

barrier_packages = concatenate_certificates(*barriers)

# Instantiate Controller
controller = cbf_controller(
    control_limits=actuation_constraints,
    dynamics_func=unicycle_dynamics,
    barriers=barrier_packages,
)

# Execute Simulation
# usage of use_jit=True ensures the entire simulation loop is JIT compiled,
# drastically improving performance by avoiding Python interpreter overhead during stepping.
x, u, z, p, dkeys, dvals, planner_data_keys, planner_data_values = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=unicycle_dynamics,
    integrator=integrator,
    planner=mppi_local_planner,  # Use MPPI as planner
    nominal_controller=None,  # No separate nominal controller (MPPI provides u_nom)
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    filepath=file_path + "mppi_cbf_results",
    verbose=True,
    planner_data=PlannerData(
        u_traj=jnp.zeros((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"])),
        x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, int(tf / dt) + 1)),
        prev_robustness=None,
    ),
    use_jit=True,
)

# Visualization
plot = 0
animate = 0
save = 1

if plot:
    import matplotlib.pyplot as plt

    from examples.unicycle.common.visualizations import plot_trajectory

    plot_trajectory(
        states=x,
        desired_state=desired_state,
        desired_state_radius=0.25,
        obstacles=obstacles,
        ellipsoids=ellipsoids,
        x_lim=(-2, 6),
        y_lim=(-2, 6),
        title="System Behavior (MPPI + CBF)",
    )
    plt.show()

if animate:
    from cbfkit.utils.visualizations.plot_mppi_ellipsoid_environment import animate as animate_func

    animate_func(
        states=x,
        estimates=z,
        controller_data_keys=planner_data_keys,
        controller_data_items=planner_data_values,
        mppi_args=mppi_args,
        desired_state=desired_state,
        desired_state_radius=0.25,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        dt=dt,
        title="System Behavior (MPPI + CBF)",
        obstacles=obstacles,
        ellipsoids=ellipsoids,
        save_animation=False,
        animation_filename=file_path + "bh_mppi_cbf_control",
    )

final_pos = x[:2, -1]
desired_pos = desired_state[:2]
dist = jnp.linalg.norm(final_pos - desired_pos)
print(f"Final Distance to Goal: {dist}")
