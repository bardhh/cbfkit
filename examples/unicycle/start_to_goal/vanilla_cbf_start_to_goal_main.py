import jax.numpy as jnp

import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
import cbfkit.simulation.simulator as sim

from cbfkit.controllers.model_based.cbf_clf_controllers import (
    vanilla_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.rectify_relative_degree import (
    rectify_relative_degree,
)

# Simulation parameters
tf = 10.0
dt = 0.01
file_path = "examples/unicycle/start_to_goal/results/"

init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
actuation_constraints = jnp.array([100.0, 100.0])  # Effectively, no control limits
sigma_matrix = 0.1 * jnp.eye(len(init_state))
sigma = lambda x: sigma_matrix

unicycle_dynamics = unicycle.plant(l=1.0)
unicycle_dynamics.a_max = actuation_constraints[0]  # α_max
unicycle_dynamics.omega_max = actuation_constraints[1]  # ω_max
unicycle_dynamics.v_max = 2.0
unicycle_dynamics.goal_tol = 0.25

uniycle_nom_controller = unicycle.controllers.proportional_controller(
    dynamics=unicycle_dynamics,
    Kp_pos=1.0,
    Kp_theta=5.0,
    desired_state=desired_state,
)

obstacles = [
    (1, 2.0, 0.0),
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
    )(
        certificate_conditions=zeroing_barriers.linear_class_k(10.0),
        obstacle=obs,
        ellipsoid=ell,
    )
    for obs, ell in zip(obstacles, ellipsoids)
]
barrier_packages = concatenate_certificates(*barriers)

# barriers = [
#     unicycle.certificate_functions.barrier_functions.obstacle_ca(
#         certificate_conditions=zeroing_barriers.linear_class_k(10.0),
#         obstacle=obs,
#         ellipsoid=ell,
#     )
#     for obs, ell in zip(obstacles, ellipsoids)
# ]
# barrier_packages = concatenate_certificates(*barriers)

controller = cbf_controller(
    control_limits=actuation_constraints,
    nominal_input=uniycle_nom_controller,
    dynamics_func=unicycle_dynamics,
    barriers=barrier_packages,
)

# Simulation imports
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator


x, u, z, p, data, data_keys = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=unicycle_dynamics,
    integrator=integrator,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    filepath=file_path + "vanilla_cbf_results",
    verbose=True,
)

plot = 1
animate = 1
save = 1

if plot:
    from examples.unicycle.start_to_goal.visualizations import plot_trajectory

    plot_trajectory(
        states=x,
        desired_state=desired_state,
        desired_state_radius=0.25,
        obstacles=obstacles,
        ellipsoids=ellipsoids,
        x_lim=(-2, 6),
        y_lim=(-2, 6),
        title="System Behavior",
    )

if animate:
    from examples.unicycle.start_to_goal.visualizations import animate

    animate(
        states=x,
        estimates=z,
        desired_state=desired_state,
        desired_state_radius=0.25,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        dt=dt,
        title="System Behavior",
        obstacles=obstacles,
        ellipsoids=ellipsoids,
        save_animation=save,
        animation_filename=file_path + "bh_vanilla_cbf_control.mp4",
    )
