import os
import jax.numpy as jnp
from jax import Array, jit, lax
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
import cbfkit.simulation.simulator as sim
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation

# from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers import (
#     vanilla_cbf_clf_qp_controller as cbf_controller,
# )
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers import (
    stochastic_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    stochastic_barrier,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.rectify_relative_degree import (
    rectify_relative_degree,
)

import cbfkit.controllers_and_planners.waypoint as single_waypoint_planner
import cbfkit.controllers_and_planners.model_based.mppi as mppi_planner

# Simulation parameters
tf = 2.5  # 5  # 10.0  # 0.0
dt = 0.01
# file_path = "examples/rss_workshop/cbf_only/results/"

file_path = os.path.dirname(os.path.abspath(__file__))
target_directory = file_path + "/results"
model_name = "cbf_mppi_028_mid"

unicycle_dynamics = unicycle.plant()
init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
actuation_constraints = jnp.array([100.0, 100.0])  # Effectively, no control limits
# sigma_matrix = 0.1 * jnp.eye(len(init_state))
sigma_matrix = 0.28 * jnp.eye(len(init_state))
sigma = lambda x: sigma_matrix

target_setpoint = single_waypoint_planner.vanilla_waypoint(target_state=desired_state)

uniycle_nom_controller = unicycle.controllers.proportional_controller(
    dynamics=unicycle_dynamics,
    Kp_pos=1.0,
    Kp_theta=10.0,
)

obstacles = [
    (1.0, 2.0, 0.0),
    (3.0, 2.0, 0.0),
    (2.0, 5.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.5, -1.0, 0.0),
]
obstacles_array = jnp.asarray(obstacles)
ellipsoids = [
    (0.5, 1.5),
    (0.75, 2.0),
    (2.0, 0.25),
    (1.0, 0.75),
    (0.75, 0.5),
]
ellipsoids_array = jnp.asarray(ellipsoids)

# barriers = [
#     rectify_relative_degree(
#         function=unicycle.certificate_functions.barrier_functions.ellipsoidal_obstacle.cbf(
#             obs, ell
#         ),
#         system_dynamics=unicycle_dynamics,
#         state_dim=len(init_state),
#         form="exponential",
#     )(certificate_conditions=zeroing_barriers.linear_class_k(1.0), obstacle=obs, ellipsoid=ell)
#     for obs, ell in zip(obstacles, ellipsoids)
# ]

barriers = [
    rectify_relative_degree(
        function=unicycle.certificate_functions.barrier_functions.ellipsoidal_obstacle.stochastic_cbf(
            obs,
            ell,
        ),
        system_dynamics=unicycle_dynamics,
        state_dim=len(init_state),
        form="exponential",
    )(
        certificate_conditions=stochastic_barrier.right_hand_side(
            alpha=1.0, beta=1.0
        ),  # 1.0, 1.0),
        obstacle=obs,
        ellipsoid=ell,
    )
    for obs, ell in zip(obstacles, ellipsoids)
]

barrier_packages = concatenate_certificates(*barriers)

# controller = cbf_controller(
#     control_limits=actuation_constraints,
#     # nominal_input=uniycle_nom_controller,
#     dynamics_func=unicycle_dynamics,
#     barriers=barrier_packages,
# )

controller = cbf_controller(
    control_limits=actuation_constraints,
    nominal_input=uniycle_nom_controller,
    dynamics_func=unicycle_dynamics,
    barriers=barrier_packages,
    sigma=sigma,
)


# MPPI costs
@jit
def stage_cost(state_and_time: Array, action: Array) -> Array:
    x_e, y_e = state_and_time[0], state_and_time[1]
    cost = 2.0 * ((x_e - desired_state[0]) ** 2 + (y_e - desired_state[1]) ** 2)
    # return cost

    def body(i, inputs):
        cost = inputs
        x_o, y_o, _ = obstacles_array[i, :]
        a1, a2 = ellipsoids_array[i, :]
        d = ((x_e - x_o) / (a1)) ** 2 + ((y_e - y_o) / (a2)) ** 2 - 1.0
        cost = cost + 2.0 / jnp.max(jnp.array([d, 0.01]))
        return cost

    cost = lax.fori_loop(0, len(obstacles), body, cost)
    return cost


@jit
def terminal_cost(state_and_time: Array, action: Array) -> Array:
    x_e, y_e = state_and_time[0], state_and_time[1]
    cost = 10.0 * ((x_e - desired_state[0]) ** 2 + (y_e - desired_state[1]) ** 2)
    return cost

    def body(i, inputs):
        cost = inputs
        x_o, y_o, _ = obstacles_array[i, :]
        a1, a2 = ellipsoids_array[i, :]
        d = ((x_e - x_o) / (a1)) ** 2 + ((y_e - y_o) / (a2)) ** 2 - 1.0
        cost = cost + 5.0 / jnp.max(jnp.array([d, 0.01]))
        return cost

    cost = lax.fori_loop(0, len(obstacles), body, cost)
    return cost


# Instantiate MPPI control law
mppi_args = {
    "robot_state_dim": 4,
    "robot_control_dim": 2,
    "prediction_horizon": 80,  # 150,
    "num_samples": 20000,
    "plot_samples": 30,
    "time_step": dt * 2.0,
    "use_GPU": False,
    "costs_lambda": 0.03,
    "cost_perturbation": 0.1,
}
mppi_local_planner = mppi_planner.vanilla_mppi(
    control_limits=actuation_constraints,
    dynamics_func=unicycle_dynamics,
    trajectory_cost=None,  # trajectory_cost,
    stage_cost=stage_cost,  # ,stage_cost,
    terminal_cost=terminal_cost,  # terminal_cost,
    mppi_args=mppi_args,
)

# Simulation imports
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator

u_guess = jnp.append(
    jnp.ones((mppi_args["prediction_horizon"], 1)),
    jnp.zeros((mppi_args["prediction_horizon"], 1)),
    axis=1,
)
(
    x,
    u,
    z,
    p,
    controller_data_keys_,
    controller_data_items_,
    planner_data_keys_,
    planner_data_items_,
) = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=unicycle_dynamics,
    integrator=integrator,
    planner=mppi_local_planner,  # target_setpoint,  # mppi_local_planner,  # None,  # ,
    nominal_controller=uniycle_nom_controller,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    perturbation=generate_stochastic_perturbation(sigma=sigma, dt=dt),
    filepath=file_path + "vanilla_cbf_results",
    planner_data={"u_traj": u_guess, "prev_robustness": None},
    controller_data={},
)

plot = 1
animate = 1
save = 0

# if plot:
#     from examples.rss_workshop.visualizations import plot_trajectory

#     plot_trajectory(
#         states=x,
#         desired_state=desired_state,
#         desired_state_radius=0.25,
#         obstacles=obstacles,
#         ellipsoids=ellipsoids,
#         x_lim=(-2, 6),
#         y_lim=(-2, 6),
#         title="System Behavior",
#         savefile="examples/rss_workshop/cbf_only/results/vanilla_cbf.png",
#     )

if animate:
    from examples.rss_workshop.visualizations import animate

    animate(
        states=x,
        estimates=z,
        controller_data_keys=planner_data_keys_,
        controller_data_items=planner_data_items_,
        mppi_args=mppi_args,
        desired_state=desired_state,
        desired_state_radius=0.1,
        x_lim=(-2, 6),
        y_lim=(-2, 6),
        dt=dt,
        title="System Behavior",
        obstacles=obstacles,
        ellipsoids=ellipsoids,
        save_animation=save,
        animation_filename=target_directory + "/" + model_name,  # + ".mp4",
    )
