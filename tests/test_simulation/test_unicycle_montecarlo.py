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

import unittest
from typing import List
import jax.numpy as jnp
from jax import jacfwd, Array
from numpy import random

from cbfkit.simulation import simulator as sim
from cbfkit.simulation import monte_carlo
from cbfkit.controllers.model_based.cbf_clf_controllers import vanilla_cbf_clf_qp_controller
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions.zeroing_barriers import (
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


def execute(_ii: int) -> List[Array]:
    """_summary_

    Args:
        int (ii): _description_

    Returns:
        List[Array]: _description_
    """
    invalid_initial_condition = True
    # Generate random initial condition
    while invalid_initial_condition:
        x_rand = random.uniform(low=-x_max, high=x_max)
        y_rand = random.uniform(low=-y_max, high=y_max)
        a_rand = random.uniform(low=-jnp.pi, high=jnp.pi)
        initial_state = jnp.array([x_rand, y_rand, a_rand])

        invalid_initial_condition = any(
            (x_rand - xo[0]) ** 2 + (y_rand - xo[1]) ** 2 - ro[0] ** 2 < 0
            for xo, ro in zip(OBSTACLES, ELLIPSOIDS)
        )

    _x, _u, _z, _p, dkeys, dvalues = sim.execute(
        x0=initial_state,
        dt=DT,
        num_steps=N_STEPS,
        dynamics=DYNAMICS,
        perturbation=generate_stochastic_perturbation(lambda _: Q, DT),
        integrator=integrator,
        controller=CONTROLLER,
        sensor=sensor,
        estimator=ESTIMATOR,
        sigma=R,
    )

    if "error" in dkeys:
        if dvalues[-1][dkeys.index("error")]:
            return False

    return True


def run_monte_carlo(n_trials: int) -> bool:
    """_summary_

    Args:
        controller (ControllerCallable): _description_

    Returns:
        bool: _description_
    """
    successes = monte_carlo.conduct_monte_carlo(
        execute,
        n_trials,
    )

    if False in successes:
        return False

    return True


class TestUnicycleMonteCarlo(unittest.TestCase):
    """Takes care of unit tests intended to simulate Monte Carlo trials of the
    unicycle start-to-goal case study.

    """

    def test_unicycle_cbf_controller_controller(self):
        """Tests a unicycle Monte-Carlo simulation with a CLF controller."""

        self.assertTrue(run_monte_carlo(N_TRIALS))


if __name__ == "__main__":
    unittest.main()
