import os
import sys

# Add the project root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import jax.numpy as jnp

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import stochastic_barrier
from cbfkit.controllers.cbf_clf import stochastic_cbf_clf_qp_controller as cbf_controller
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
from cbfkit.sensors import perfect as sensor
from cbfkit.utils.user_types import PlannerData
from examples.unicycle.common.ellipsoidal_obstacle import stochastic_cbf as ellipsoid_cbf
from examples.unicycle.common.visualizations import animate, plot_trajectory

plot = 1 if not os.getenv("CBFKIT_TEST_MODE") else 0
should_animate = 1 if not os.getenv("CBFKIT_TEST_MODE") else 0
save = 0

# Simulation parameters
tf = 10.0 if not os.getenv("CBFKIT_TEST_MODE") else 0.5
dt = 0.01
file_path = "examples/unicycle/reach_goal/results/"

unicycle_dynamics = unicycle.plant()
init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
actuation_constraints = jnp.array([100.0, 100.0])  # Effectively, no control limits
sigma_matrix = 0.1 * jnp.eye(len(init_state))


def sigma(x):
    return sigma_matrix


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
ellipsoids = [
    (0.5, 1.5),
    (0.75, 2.0),
    (2.0, 0.25),
    (1.0, 0.75),
    (0.75, 0.5),
]

barriers = [
    rectify_relative_degree(
        function=ellipsoid_cbf(
            jnp.array(obs),
            jnp.array(ell),
        ),
        system_dynamics=unicycle_dynamics,
        state_dim=len(init_state),
        form="exponential",
    )(
        certificate_conditions=stochastic_barrier.right_hand_side(alpha=1.0, beta=1.0),
    )
    for obs, ell in zip(obstacles, ellipsoids)
]

barrier_packages = concatenate_certificates(*barriers)

controller = cbf_controller(
    control_limits=actuation_constraints,
    nominal_input=uniycle_nom_controller,
    dynamics_func=unicycle_dynamics,
    barriers=barrier_packages,
    sigma=sigma,
    relaxable_cbf=True,
)

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
    perturbation=generate_stochastic_perturbation(sigma=sigma, dt=dt),
    filepath=file_path + "stochastic_cbf_results",
    planner_data=PlannerData(x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, int(tf / dt) + 1))),
)

if plot:
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

if should_animate:  # Changed from if animate:
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
        animation_filename=file_path + "stochastic_cbf_control.mp4",
    )
