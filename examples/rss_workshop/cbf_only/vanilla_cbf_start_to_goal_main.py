import jax.numpy as jnp

import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
import cbfkit.simulation.simulator as sim
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers import (
    vanilla_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.rectify_relative_degree import (
    rectify_relative_degree,
)

# Simulation parameters
tf = 10.0
dt = 0.01
file_path = "examples/rss_workshop/cbf_only/results/"

unicycle_dynamics = unicycle.plant()
init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
actuation_constraints = jnp.array([100.0, 100.0])  # Effectively, no control limits
sigma_matrix = 0.1 * jnp.eye(len(init_state))
sigma = lambda x: sigma_matrix

import cbfkit.controllers_and_planners.waypoint as single_waypoint_planner

target_setpoint = single_waypoint_planner.vanilla_waypoint(target_state=desired_state)

uniycle_nom_controller = unicycle.controllers.proportional_controller(
    dynamics=unicycle_dynamics,
    Kp_pos=1.0,
    Kp_theta=10.0,
    desired_state=desired_state,
)

obstacles = [
    (1.0, 2.0, 0.0),
    (3.0, 2.0, 0.0),
    (2.0, 5.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.5, -1.0, 0.0),
]
ellipsoids = [
    (0.5, 1.5),
    (0.75, 2.0),
    (2.0, 0.25),
    (1.0, 0.75),
    (0.75, 0.5),
]

# cbf_package = rectify_relative_degree(
#     function=unicycle.certificate_functions.barrier_functions.ellipsoidal_obstacle.cbf(),
#     system_dynamics=unicycle_dynamics,
#     state_dim=len(init_state),
#     form="exponential",
# )

barriers = [
    rectify_relative_degree(
        function=unicycle.certificate_functions.barrier_functions.ellipsoidal_obstacle.cbf(
            obs, ell
        ),
        system_dynamics=unicycle_dynamics,
        state_dim=len(init_state),
        form="exponential",
    )(certificate_conditions=zeroing_barriers.linear_class_k(1.0), obstacle=obs, ellipsoid=ell)
    for obs, ell in zip(obstacles, ellipsoids)
]
# rectify_relative_degree(
#     function=unicycle.certificate_functions.barrier_functions.ellipsoidal_obstacle.cbf(),
#     system_dynamics=unicycle_dynamics,
#     state_dim=len(init_state),
#     form="exponential",
# )

# barriers = [
#     unicycle.certificate_functions.barrier_functions.obstacle_ca(
#         certificate_conditions=zeroing_barriers.linear_class_k(2.0),
#         obstacle=obs,
#         ellipsoid=ell,
#     )
#     for obs, ell in zip(obstacles, ellipsoids)
# ]
barrier_packages = concatenate_certificates(*barriers)

controller = cbf_controller(
    control_limits=actuation_constraints,
    # nominal_input=uniycle_nom_controller,
    dynamics_func=unicycle_dynamics,
    barriers=barrier_packages,
)

# Simulation imports
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator


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
    planner=target_setpoint,
    nominal_controller=uniycle_nom_controller,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    perturbation=generate_stochastic_perturbation(sigma=sigma, dt=dt),
    filepath=file_path + "vanilla_cbf_results",
)

plot = 1
animate = 1
save = 0

if plot:
    from examples.rss_workshop.visualizations import plot_trajectory

    plot_trajectory(
        states=x,
        desired_state=desired_state,
        desired_state_radius=0.25,
        obstacles=obstacles,
        ellipsoids=ellipsoids,
        x_lim=(-2, 6),
        y_lim=(-2, 6),
        title="System Behavior",
        savefile="examples/rss_workshop/cbf_only/results/vanilla_cbf.png",
    )

if animate:
    from examples.rss_workshop.visualizations import animate

    animate(
        states=x,
        estimates=z,
        desired_state=desired_state,
        desired_state_radius=0.1,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        dt=dt,
        title="System Behavior",
        obstacles=obstacles,
        ellipsoids=ellipsoids,
        save_animation=save,
        animation_filename=file_path + "vanilla_cbf_control.gif",
    )
