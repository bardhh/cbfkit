"""
This module provides a template for example simulations designed to be run from
the interpreter via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the
src/cbfkit tree.

"""

import jax.numpy as jnp
import numpy as np
from jax import jacfwd

# Whether or not to simulate, plot
simulate = 1
plot = 1
save = 1

# Load system module
import cbfkit.simulation.simulator as sim

# Load dynamics, sensors, controller, estimator, integrator
from cbfkit.systems import unicycle
from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.estimators import ct_hybrid_ekf_ukf_dtmeas as hybrid_ekf_ukf
from cbfkit.integration import forward_euler as integrator

# Load initial conditions
from examples.unicycle.common.config import (
    ekf_ukf_hybrid_state_estimation as initial_conditions,
)

# Define time parameters
tf = 5.0
dt = 0.01
n_steps = int(tf / dt)

# Define dynamics, controller, and estimator with specified parameters
dynamics = unicycle.approx_unicycle_dynamics(l=1.0)
dfdx = lambda x: jacfwd(dynamics)(x)
h = lambda x: x
dhdx = lambda x: jnp.eye((len(initial_conditions.initial_state)))
controller = unicycle.proportional_controller(dynamics=dynamics, Kp_pos=1, Kp_theta=0.01)
scale_factor = 1.25
estimator = hybrid_ekf_ukf(
    Q=initial_conditions.Q * scale_factor,
    R=initial_conditions.R * scale_factor,
    dynamics=dynamics,
    dfdx=dfdx,
    h=h,
    dhdx=dhdx,
    dt=dt,
)

if simulate:
    # Execute simulation
    # Compute initial covariance for hybrid estimator (P = [Pe; Pu])
    R_scaled = initial_conditions.R * scale_factor
    initial_covariance = jnp.vstack([R_scaled, R_scaled])

    (
        states,
        controls,
        estimates,
        covariances,
        data_keys,
        data_values,
        planner_data,
        planner_data_keys,
    ) = sim.execute(
        x0=initial_conditions.initial_state,
        dynamics=dynamics,
        sensor=sensor,
        nominal_controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=dt,
        sigma=initial_conditions.R,
        num_steps=n_steps,
        planner_data={
            "u_traj": None,
            "x_traj": jnp.tile(initial_conditions.desired_state.reshape(-1, 1), (1, n_steps + 1)),
            "prev_robustness": None,
        },
        initial_covariance=initial_covariance,
        use_jit=True,
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
    from examples.unicycle.common.visualizations import animate

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
        animation_filename="examples/unicycle/reach_goal/results/ekf_ukf_hybrid_estimation.gif",
    )
