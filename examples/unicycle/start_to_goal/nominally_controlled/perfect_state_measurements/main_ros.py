"""
This module provides a template for example simulations designed to be run from
the interpreter via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the
src/cbfkit tree.

"""

import jax.numpy as jnp
import numpy as np

# Whether or not to simulate, plot
ros = 1
simulate = 0
view_only = (not ros) and (not simulate)

# Plot or save
plot = 1
save = 0

# Load initial conditions
from examples.unicycle.start_to_goal.nominally_controlled.perfect_state_measurements import (
    initial_conditions,
)

if simulate:
    # Load simulation module
    import cbfkit.system as system

    # Load dynamics, sensors, integrator
    from cbfkit.sensors import perfect as sensor
    from cbfkit.integration import forward_euler as integrator

elif ros:
    #! TO DO
    # # Load ROS simulation module
    import cbfkit.ros.system as system
    import cbfkit.ros.sensor as sensor
    from cbfkit.models.unicycle import proportional_controller

# Load controller and estimator
from cbfkit.models.unicycle import approx_unicycle_dynamics, proportional_controller
from cbfkit.estimators import naive as estimator

# Define dynamics and controller with specified parameters
approx_unicycle_dynamics = approx_unicycle_dynamics(l=1.0, sigma=initial_conditions.Q)
controller = proportional_controller(
    dynamics=approx_unicycle_dynamics,
    Kp_pos=1,
    Kp_theta=0.01,
    desired_state=initial_conditions.desired_state,
)

# Define time parameters
tf = 5.0
dt = 0.01
n_steps = int(tf / dt)

if not view_only:
    # Execute simulation
    states, controls, estimates, covariances, data_keys, data_values = system.simulate(
        x0=initial_conditions.initial_state,
        dynamics=approx_unicycle_dynamics,
        sensor=sensor,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=dt,
        R=initial_conditions.R,
        num_steps=n_steps,
    )

    # Reformat results as numpy arrays
    states = np.array(states)
    controls = np.array(controls)
    estimates = np.array(estimates)
    covariances = np.array(covariances)

else:
    #! TO DO: Implement load from file
    pass

if plot:
    from examples.unicycle.start_to_goal.visualizations import animate

    animate(
        states=states,
        estimates=estimates,
        desired_state=initial_conditions.desired_state,
        desired_state_radius=0.1,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        dt=dt,
        title="System Behavior",
        save_animation=save,
        animation_filename="examples/unicycle/start_to_goal/nominally_controlled/perfect_state_estimation/results/test.gif",
    )
