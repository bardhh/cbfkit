"""
Test Module for conducting a Monte Carlo simulation of the unicycle system
=========================

This module contains tests that simulate the unicycle system under various
controllers for 100 trials each.

Tests
-----
- test_unicycle_start_to_goal: reach the goal set

Setup
-----
- No set up required

Examples
--------
To run all tests in this module (from the root of the repository):
    $ python -m unittest tests.test_simulation.test_fixedwing_simulation.py
"""

import time
import unittest
import jax.numpy as jnp
from jax import jacfwd, random
from numpy.random import uniform

from cbfkit.simulation.simulator import stepper
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.vanilla_cbf_clf_qp_control_laws import (
    vanilla_cbf_clf_qp_controller,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.barrier_conditions.zeroing_barriers import (
    linear_class_k,
)
from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.estimators import ct_ekf_dtmeas
from cbfkit.integration import forward_euler as integrator
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation


# Simulation-specific
import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle

# Simulation setup
N = 3  # n_states
M = 2  # n_controls
TF = 5.0
DT = 1e-2
N_STEPS = int(TF / DT)
N_TRIALS = 4

# Initial conditions
x_max = 5.0
y_max = 5.0
ACTUATION_LIMITS = jnp.array([1e3, 1e3])

# Stochastic noise parameters
Q = 0.5 * jnp.eye(N)  # process noise
R = 0.05 * jnp.eye(N)  # measurement noise

# Barrier function parameters
ALPHA = 0.5
GOAL = jnp.array([0.0, 0.0, 0])
RAD = 0.05
OBSTACLES = [
    jnp.array([1.0, 2.0]),
    jnp.array([2.0, 2.0]),
    jnp.array([0.0, 3.0]),
]
ELLIPSOIDS = [
    jnp.array([0.5, 0.5]),
    jnp.array([0.5, 0.5]),
    jnp.array([0.5, 0.5]),
]

# Lyapunov function
bars = [
    unicycle.certificate_functions.barrier_functions.obstacle_ca(
        certificate_conditions=linear_class_k(ALPHA),
        obstacle=jnp.array([obs[0], obs[1], 0.0]),
        ellipsoid=jnp.array([ell[0], ell[1]]),
    )
    for obs, ell in zip(OBSTACLES, ELLIPSOIDS)
]
BARRIERS = concatenate_certificates(*bars)

DYNAMICS = unicycle.plant(l=1.0)
NOMINAL_CONTROLLER = unicycle.controllers.proportional_controller(
    dynamics=DYNAMICS,
    Kp_pos=1.0,
    Kp_theta=0.01,
    desired_state=GOAL,
)
DFDX = jacfwd(DYNAMICS)
H = lambda x: x
DHDX = lambda _x: jnp.eye(N)
ESTIMATOR = ct_ekf_dtmeas(
    Q=Q,
    R=R,
    dynamics=DYNAMICS,
    dfdx=DFDX,
    h=H,
    dhdx=DHDX,
    dt=DT,
)
CONTROLLER = vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    nominal_input=NOMINAL_CONTROLLER,
    dynamics_func=DYNAMICS,
    barriers=BARRIERS,
)

step = stepper(
    dt=DT,
    dynamics=DYNAMICS,
    perturbation=generate_stochastic_perturbation(sigma=lambda _x: Q, dt=DT),
    integrator=integrator,
    controller=CONTROLLER,
    sensor=sensor,
    estimator=ESTIMATOR,
    sigma=R,
    key=random.PRNGKey(0),
)


class TestControlLoop(unittest.TestCase):
    """Takes care of unit tests intended to examine the performance of a control loop."""

    def test_loop_less_than_10ms(self):
        """Tests to make sure that the average loop time is less than 10ms."""
        invalid_initial_condition = True
        # Generate random initial condition
        while invalid_initial_condition:
            x_rand = uniform(low=-x_max, high=x_max)
            y_rand = uniform(low=-y_max, high=y_max)
            a_rand = uniform(low=-jnp.pi, high=jnp.pi)
            initial_state = jnp.array([x_rand, y_rand, a_rand])

            invalid_initial_condition = any(
                (x_rand - xo[0]) ** 2 + (y_rand - xo[1]) ** 2 - ro[0] ** 2 < 0
                for xo, ro in zip(OBSTACLES, ELLIPSOIDS)
            )

        # Set up initial conditions
        x = initial_state
        u = jnp.zeros((DYNAMICS(x)[1].shape[1],))
        z = 0 * initial_state
        p = jnp.zeros((z.shape[0], z.shape[0]))

        elapsed_times = jnp.zeros((N_STEPS,))
        for ii in range(N_STEPS):
            start_time = time.time()

            # Take one step
            x, u, z, p, _ = step(DT * ii, x, u, z, p)

            elapsed_times = elapsed_times.at[ii].set(time.time() - start_time)

        mean_elapsed_ms = 1000 * jnp.mean(jnp.array(elapsed_times))
        mean_elapsed_ms_no_first = 1000 * jnp.mean(jnp.array(elapsed_times[1:]))
        print(f"Mean Elapsed (ms): {mean_elapsed_ms:.2f}")
        print(f"Mean Elapsed - No First (ms): {mean_elapsed_ms_no_first:.2f}")

        self.assertTrue(mean_elapsed_ms_no_first < 10.0)


if __name__ == "__main__":
    unittest.main()
