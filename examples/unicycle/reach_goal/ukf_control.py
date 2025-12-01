"""This module provides a template for example simulations designed to be run from the interpreter
via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the src/cbfkit tree.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array, jacfwd

import cbfkit.simulation.simulator as sim
from cbfkit.estimators import ct_ukf_dtmeas as ukf
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.systems import unicycle
from cbfkit.utils.user_types import PlannerData
from examples.unicycle.common.config import ukf_state_estimation as initial_conditions
from examples.unicycle.common.visualizations import animate

# Whether or not to simulate, plot
simulate = 1
plot = 1
save = 1

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


controller = unicycle.proportional_controller(dynamics=dynamics, Kp_pos=1, Kp_theta=0.01)
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
    initial_state = jnp.array(initial_conditions.initial_state)

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
        x0=initial_state,
        dynamics=dynamics,
        sensor=sensor,
        nominal_controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=dt,
        sigma=initial_conditions.R,
        num_steps=n_steps,
        planner_data=PlannerData(
            u_traj=None,
            x_traj=jnp.tile(initial_conditions.desired_state.reshape(-1, 1), (1, n_steps + 1)),
            prev_robustness=None,
        ),
        use_jit=True,
    )

else:
    # Implement load from file
    pass

if plot:
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
        animation_filename="examples/unicycle/reach_goal/results/ukf_estimation.gif",
    )
