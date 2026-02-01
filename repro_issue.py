import jax.numpy as jnp
from jax import Array, jit
import cbfkit.controllers.mppi as mppi_planner
import cbfkit.simulation.simulator as sim
from cbfkit.systems.unicycle.models.accel_unicycle import plant
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator

# Simulation parameters
tf, dt = 10.0, 0.05
init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
actuation_constraints = jnp.array([100.0, 100.0])

unicycle_dynamics = plant(lam=1.0)

# MPPI Cost Functions
@jit
def stage_cost(state_and_time: Array, action: Array) -> Array:
    x, y, xd, yd = state_and_time[0], state_and_time[1], desired_state[0], desired_state[1]
    return 10.0 * ((x - xd) ** 2 + (y - yd) ** 2)

@jit
def terminal_cost(state_and_time: Array, action: Array) -> Array:
    x, y, xd, yd = state_and_time[0], state_and_time[1], desired_state[0], desired_state[1]
    return 100.0 * ((x - xd) ** 2 + (y - yd) ** 2)

# MPPI Configuration
mppi_args = {
    "robot_state_dim": 4,
    "robot_control_dim": 2,
    "prediction_horizon": 50,
    "num_samples": 500,
    "plot_samples": 30,
    "time_step": dt,
    "use_GPU": False,
    "costs_lambda": 0.1,
    "cost_perturbation": 0.5,
}

# Instantiate MPPI Planner
mppi_local_planner = mppi_planner.vanilla_mppi(
    control_limits=actuation_constraints,
    dynamics_func=unicycle_dynamics,
    trajectory_cost=None,
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    mppi_args=mppi_args,
)

# Define Obstacles and Generate Barriers
obstacles, ellipsoids = [(1.0, 2.0, 0.0)], [(0.5, 1.5)]
cbf_factory, _, _ = ellipsoidal_barrier_factory(
    system_position_indices=(0, 1), obstacle_position_indices=(0, 1), ellipsoid_axis_indices=(0, 1)
)

barriers = [
    rectify_relative_degree(
        function=cbf_factory(obs, ell),
        system_dynamics=unicycle_dynamics,
        state_dim=len(init_state),
        form="exponential",
    )(certificate_conditions=zeroing_barriers.linear_class_k(5.0))
    for obs, ell in zip(obstacles, ellipsoids)
]

controller = cbf_controller(
    control_limits=actuation_constraints,
    dynamics_func=unicycle_dynamics,
    barriers=concatenate_certificates(*barriers),
)

# Simulation Execution
x, u, z, p, dkeys, dvals, planner_data, planner_data_keys = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=unicycle_dynamics,
    integrator=integrator,
    planner=mppi_local_planner,
    nominal_controller=None,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    filepath="mppi_cbf_results",
    verbose=True,
    planner_data={
        "u_traj": jnp.zeros((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"])),
        "x_traj": jnp.tile(desired_state.reshape(-1, 1), (1, int(tf / dt) + 1)),
        "prev_robustness": None,
    },
    use_jit=True,
)
