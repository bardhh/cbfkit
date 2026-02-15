"""
This module provides a template for example simulations designed to be run from
the interpreter via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the
src/cbfkit tree.

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))



import jax.numpy as jnp
import numpy as np
from jax import jacfwd

# Whether or not to simulate, plot
plot = 1
save = 1
save_path = "examples/unicycle/reach_goal/results/"  # nominally_controlled/ekf_state_estimation/results/"
file_name = os.path.basename(__file__)[:-8]

# Load system module
import cbfkit.simulation.simulator as sim

# Load dynamics, sensors, controller, estimator, integrator
import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
from cbfkit.estimators import ct_ukf_dtmeas as ukf
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
from cbfkit.sensors import unbiased_gaussian_noise as sensor

# Load initial conditions
from examples.unicycle.common.config import (
    ukf_state_estimation as initial_conditions,
)

# Define time parameters
tf = 5.0
dt = 0.01
n_steps = int(tf / dt)

# Define dynamics, controller, and estimator with specified parameters
dynamics = unicycle.plant(lam=1.0)


def dfdx(x):
    return jacfwd(unicycle.plant)(x)


def h(x):
    return x


def dhdx(x):
    return jnp.eye((len(initial_conditions.initial_state)))


controller = unicycle.controllers.proportional_controller(
    dynamics=dynamics,
    Kp_pos=1.0,
    Kp_theta=1.0,
)
scale_factor = 1.25
estimator = ukf(
    Q=initial_conditions.Q * scale_factor,
    R=initial_conditions.R * scale_factor,
    dynamics=dynamics,
    h=h,
    dt=dt,
)

# Execute simulation
x, u, z, p, data, data_keys, planner_data, planner_data_keys = sim.execute(
    x0=initial_conditions.initial_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=dynamics,
    integrator=integrator,
    nominal_controller=controller,
    sensor=sensor,
    estimator=estimator,
    perturbation=generate_stochastic_perturbation(sigma=lambda x: initial_conditions.Q, dt=dt),
    sigma=initial_conditions.R,
    filepath=save_path + file_name,
    planner_data={
        "u_traj": None,
        "x_traj": jnp.tile(initial_conditions.desired_state.reshape(-1, 1), (1, int(tf / dt) + 1)),
        "prev_robustness": None,
    },
)

if plot:
    from examples.unicycle.common.visualizations import animate

    animate(
        states=x,
        estimates=z,
        desired_state=initial_conditions.desired_state,
        desired_state_radius=0.1,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        dt=dt,
        title="System Behavior",
        save_animation=save,
        animation_filename=save_path + file_name + ".mp4",
    )
