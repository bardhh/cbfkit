"""
This module provides a template for example simulations designed to be run from
the interpreter via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the
src/cbfkit tree.

"""

import jax.numpy as jnp
import numpy as np

# Whether or not to simulate, plot
simulate = 1
plot = 0

# Load system module
import cbfkit.system as system

# Load dynamics, sensors, controller, estimator, integrator
from cbfkit.models import unicycle
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator

# Define dynamics and controller with specified parameters
approx_unicycle_dynamics = unicycle.approx_unicycle_dynamics(l=1.0, stochastic=True)
controller = unicycle.zero_controller()

# Define (semi-random) initial conditions
desired_state = jnp.array([0.0, 0.0, 0])
x_max = 5.0
y_max = 5.0
x_rand = np.random.uniform(low=-x_max, high=x_max)
y_rand = np.random.uniform(low=-y_max, high=y_max)
a_rand = jnp.arctan2(desired_state[1] - y_rand, desired_state[0] - x_rand) + np.random.uniform(
    low=-jnp.pi / 4, high=jnp.pi / 4
)
initial_state = jnp.array([x_rand, y_rand, a_rand])

# Define time parameters
tf = 2.0
dt = 0.01
n_steps = int(tf / dt)

if simulate:
    # Execute simulation
    states, controls, estimates, covariances, data_keys, data_values = system.simulate(
        x0=initial_state,
        dynamics=approx_unicycle_dynamics,
        sensor=sensor,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=dt,
        num_steps=n_steps,
    )

    # Reformat results as numpy arrays
    states = np.array(states)
    controls = np.array(controls)
    estimates = np.array(estimates)
    covariances = np.array(covariances)

else:
    # Implement load from file
    pass

if plot:
    # import pickle
    # import matplotlib.pyplot as plt

    # Implement here
    pass
