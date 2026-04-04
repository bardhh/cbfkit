"""Unicycle reach-goal with perfect state measurements and proportional control."""

import os
import sys

# Add the project root to the path so we can import examples
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_path)


import jax.numpy as jnp
import numpy as np
from jax import Array, random

import cbfkit.simulation.simulator as sim
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems import unicycle
from cbfkit.utils.user_types import PlannerData
from examples.unicycle.common.config import perfect_state_measurements as initial_conditions

# Define dynamics and controller with specified parameters
approx_unicycle_dynamics = unicycle.approx_unicycle_dynamics(lam=1.0)
controller = unicycle.proportional_controller(
    dynamics=approx_unicycle_dynamics,
    Kp_pos=1,
    Kp_theta=0.01,
)

# Define time parameters
tf = 5.0 if not os.getenv("CBFKIT_TEST_MODE") else 0.5
dt = 0.01
n_steps = int(tf / dt)

# Whether or not to simulate
simulate = 1

# Plot or save
plot = 1 if not os.getenv("CBFKIT_TEST_MODE") else 0
save = 0

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
        dynamics=approx_unicycle_dynamics,
        sensor=sensor,
        nominal_controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=dt,
        sigma=jnp.array(initial_conditions.R),
        num_steps=n_steps,
        planner_data=PlannerData(
            u_traj=None,
            x_traj=jnp.tile(initial_conditions.desired_state.reshape(-1, 1), (1, n_steps + 1)),
            prev_robustness=None,
        ),
    )

else:
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
        animation_filename="examples/unicycle/reach_goal/results/perfect_sensing.gif",
    )
