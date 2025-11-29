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
from jax import Array, jacfwd, random

# Simulation-specific
import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
from cbfkit.certificates import concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.estimators import ct_ekf_dtmeas
from cbfkit.integration import forward_euler as integrator
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.simulation import monte_carlo
from cbfkit.simulation import simulator as sim
from cbfkit.utils.user_types import PlannerData

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
ALPHA = 0.1
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
    unicycle.certificates.barrier_functions.obstacle_ca(
        certificate_conditions=linear_class_k(ALPHA),
        obstacle=jnp.array([obs[0], obs[1], 0.0]),
        ellipsoid=jnp.array([ell[0], ell[1]]),
    )
    for obs, ell in zip(OBSTACLES, ELLIPSOIDS)
]
BARRIERS = concatenate_certificates(*bars)

DYNAMICS = unicycle.plant(lam=1.0)
NOMINAL_CONTROLLER = unicycle.controllers.proportional_controller(
    dynamics=DYNAMICS,
    Kp_pos=1.0,
    Kp_theta=0.01,
)
DFDX = jacfwd(DYNAMICS)


def H(x):
    return x


def DHDX(_x):
    return jnp.eye(N)


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
    dynamics_func=DYNAMICS,
    barriers=BARRIERS,
    relaxable_clf=True,
    relaxable_cbf=True,
    slack_bound_cbf=1e10,
    slack_penalty_cbf=1.0,
)


def execute(_ii: int, use_jit: bool = False) -> bool:
    """_summary_

    Args:
        int (ii): _description_

    Returns:
        List[Array]: _description_
    """
    invalid_initial_condition = True
    key = random.PRNGKey(_ii)
    # Generate random initial condition
    while invalid_initial_condition:
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        x_rand = random.uniform(subkey1, shape=(), minval=-x_max, maxval=x_max)
        y_rand = random.uniform(subkey2, shape=(), minval=-y_max, maxval=y_max)
        a_rand = random.uniform(subkey3, shape=(), minval=-jnp.pi, maxval=jnp.pi)
        initial_state = jnp.array([x_rand, y_rand, a_rand])

        invalid_initial_condition = any(
            (x_rand - xo[0]) ** 2 + (y_rand - xo[1]) ** 2 - ro[0] ** 2 < 0
            for xo, ro in zip(OBSTACLES, ELLIPSOIDS)
        )

    _x, _u, _z, _p, dkeys, dvalues, _planner_data, _planner_data_keys = sim.execute(
        x0=initial_state,
        dt=DT,
        num_steps=N_STEPS,
        dynamics=DYNAMICS,
        perturbation=generate_stochastic_perturbation(lambda _: Q, DT),
        integrator=integrator,
        nominal_controller=NOMINAL_CONTROLLER,
        controller=CONTROLLER,
        sensor=sensor,
        estimator=ESTIMATOR,
        sigma=R,
        planner_data=PlannerData(
            u_traj=None,
            x_traj=jnp.tile(GOAL.reshape(-1, 1), (1, N_STEPS + 1)),
            prev_robustness=None,
        ),
        use_jit=use_jit,
    )

    if "error" in dkeys:
        err_idx = dkeys.index("error")
        # Check if any step reported an error
        if any(step_vals[err_idx] for step_vals in dvalues):
            return False

    return True


def run_monte_carlo(n_trials: int, use_jit: bool = False) -> bool:
    """_summary_

    Args:
        controller (ControllerCallable): _description_

    Returns:
        bool: _description_
    """
    # Partial application to pass use_jit
    from functools import partial

    execute_fn = partial(execute, use_jit=use_jit)

    successes = monte_carlo.conduct_monte_carlo(
        execute_fn,
        n_trials,
        n_processes=1,  # Run sequentially to avoid JAX pickling issues
    )

    if False in successes:
        return False

    return True


class TestUnicycleMonteCarlo(unittest.TestCase):
    """Takes care of unit tests intended to simulate Monte Carlo trials of the
    unicycle start-to-goal case study.

    """

    def test_unicycle_cbf_controller_controller(self):
        """Tests a unicycle Monte-Carlo simulation with a CLF controller (Python Loop)."""
        self.assertTrue(run_monte_carlo(N_TRIALS, use_jit=False))

    def test_unicycle_cbf_controller_jit(self):
        """Tests a unicycle Monte-Carlo simulation with a CLF controller (JIT Scan)."""
        self.assertTrue(run_monte_carlo(N_TRIALS, use_jit=True))


if __name__ == "__main__":
    unittest.main()
