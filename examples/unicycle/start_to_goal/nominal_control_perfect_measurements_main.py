"""
This module provides a template for example simulations designed to be run from
the interpreter via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the
src/cbfkit tree.

"""

import os
import jax.numpy as jnp
import numpy as np
from jax import jacfwd

# Whether or not to simulate, plot
plot = 1
save = 1
save_path = "examples/unicycle/start_to_goal/results/"  # nominally_controlled/ekf_state_estimation/results/"
file_name = os.path.basename(__file__)[:-8]

# Load system module
import cbfkit.simulation.simulator as sim

# Load dynamics, sensors, controller, estimator, integrator
import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle

from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator

# Load initial conditions
from examples.unicycle.start_to_goal.initial_conditions import (
    perfect_state_measurements as initial_conditions,
)

# Define time parameters
tf = 5.0
dt = 0.01
n_steps = int(tf / dt)

# Define dynamics, controller, and estimator with specified parameters
approx_unicycle_dynamics = unicycle.plant(l=1.0)
controller = unicycle.controllers.proportional_controller(
    dynamics=approx_unicycle_dynamics,
    Kp_pos=1,
    Kp_theta=1.0,
    desired_state=initial_conditions.desired_state,
)

# Execute simulation
x, u, z, p, data, data_keys = sim.execute(
    x0=initial_conditions.initial_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=approx_unicycle_dynamics,
    integrator=integrator,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    filepath=save_path + file_name,
)

if plot:
    from examples.unicycle.start_to_goal.visualizations import animate

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
        animation_filename=save_path + file_name + ".gif",
    )
