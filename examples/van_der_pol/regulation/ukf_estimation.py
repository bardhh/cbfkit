"""
This module provides a template for example simulations designed to be run from
the interpreter via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the
src/cbfkit tree.

"""
import matplotlib

# Hack to prevent matplotlib.use("macosx") error in imported modules
matplotlib.use = lambda *args, **kwargs: None

import jax.numpy as jnp
import numpy as np
from jax import jacfwd

# Whether or not to simulate, plot
simulate = 1
plot = 1
save = 1

# Load system module
import cbfkit.simulation.simulator as system
from cbfkit.estimators.kalman_filters.ukf import ct_ukf_dtmeas as ukf
from cbfkit.integration import runge_kutta_4 as integrator

# Load dynamics, sensors, controller, estimator, integrator
from cbfkit.systems import unicycle
from cbfkit.sensors import unbiased_gaussian_noise as sensor

# Load initial conditions
from examples.van_der_pol.common.config import ukf_state_estimation as initial_conditions

# Define time parameters
tf = 5.0
dt = 0.01
n_steps = int(tf / dt)

# Define dynamics, controller, and estimator with specified parameters
dynamics = unicycle.approx_unicycle_dynamics(lam=1.0)


def dfdx(x):
    return jacfwd(dynamics)(x)


def h(x):
    return x


def dhdx(x):
    return jnp.eye((len(initial_conditions.initial_state)))


_controller = unicycle.proportional_controller(dynamics=dynamics, Kp_pos=1, Kp_theta=0.01)


def controller(t, x, u_nom, key, data):
    u, _ = _controller(t, x, key, initial_conditions.desired_state)
    return u, data


scale_factor = 1.25
estimator = ukf(
    Q=initial_conditions.Q * scale_factor,
    R=initial_conditions.R * scale_factor,
    dynamics=dynamics,
    h=h,
    dt=dt,
)


if simulate:
    # Execute simulation
    (
        states,
        controls,
        estimates,
        covariances,
        data_keys,
        data_values,
        planner_keys,
        planner_values,
    ) = system.execute(
        x0=initial_conditions.initial_state,
        dynamics=dynamics,
        sensor=sensor,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=dt,
        sigma=initial_conditions.R,
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
    import os
    from examples.unicycle.common.visualizations import animate

    if save:
        os.makedirs("examples/van_der_pol/regulation/results", exist_ok=True)

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
        animation_filename="examples/van_der_pol/regulation/results/ukf_estimation.gif",
    )
