import jax.numpy as jnp

import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
import cbfkit.simulation.simulator as sim

from cbfkit.controllers.cbf_clf import (
    vanilla_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.certificates import (
    concatenate_certificates,
    rectify_relative_degree,
)
from cbfkit.certificates.conditions.barrier_conditions import (
    zeroing_barriers,
)

from cbfkit.utils.user_types import PlannerData
from examples.unicycle.common.ellipsoidal_obstacle import cbf as ellipsoid_cbf

# Simulation parameters
tf = 10.0
dt = 0.01
file_path = "examples/unicycle/reach_goal/results/"

init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
actuation_constraints = jnp.array([100.0, 100.0])  # Effectively, no control limits

unicycle_dynamics = unicycle.plant(l=1.0)
unicycle_dynamics.a_max = actuation_constraints[0]  # α_max
unicycle_dynamics.omega_max = actuation_constraints[1]  # ω_max
unicycle_dynamics.v_max = 1.0
unicycle_dynamics.goal_tol = 0.25

uniycle_nom_controller = unicycle.controllers.proportional_controller(
    dynamics=unicycle_dynamics,
    Kp_pos=1.0,
    Kp_theta=1.0,
)

obstacles = [
    (1, 2.0, 0.0),
    (3.0, 2.0, 0.0),
    # (2.0, 5.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.5, -1.0, 0.0),
]
ellipsoids = [
    (0.5, 1.5),
    (0.75, 2.0),
    # (2.0, 0.25),
    (1.0, 0.75),
    (0.75, 0.5),
]

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

# barriers = [
#     unicycle.certificates.barrier_functions.obstacle_ca(
#         certificate_conditions=zeroing_barriers.linear_class_k(10.0),
#         obstacle=obs,
#         ellipsoid=ell,
#     )
#     for obs, ell in zip(obstacles, ellipsoids)
# ]
barrier_packages = concatenate_certificates(*barriers)
# barrier_packages = concatenate_certificates(*barriers)

controller = cbf_controller(
    control_limits=actuation_constraints,
    dynamics_func=unicycle_dynamics,
    barriers=barrier_packages,
)

# Simulation imports
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator


x, u, z, p, dkeys, dvals, planner_data, planner_data_keys = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=unicycle_dynamics,
    integrator=integrator,
    nominal_controller=uniycle_nom_controller,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    filepath=file_path + "vanilla_cbf_results",
    verbose=True,
    planner_data=PlannerData(
        x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, int(tf / dt) + 1)),
        prev_robustness=None,
    ),
    use_jit=True,
)

plot = 1
animate = 0
save = 1

if plot:
    from examples.unicycle.common.visualizations import plot_trajectory

    import matplotlib.pyplot as plt

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
    plt.show()

if animate:
    from examples.unicycle.common.visualizations import animate

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
        save_animation=False,  # Disabled due to Pillow format limitations
        animation_filename=file_path + "bh_vanilla_cbf_control.mp4",
    )

final_pos = x[:2, -1]
desired_pos = desired_state[:2]
dist = jnp.linalg.norm(final_pos - desired_pos)
print(f"Final Distance to Goal: {dist}")
