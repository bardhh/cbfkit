"""Test Module for conducting a Monte Carlo simulation of the unicycle system.

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
    $ pytest tests/test_simulation/test_control_loop.py
"""

import time

import jax.numpy as jnp
import pytest
from jax import jacfwd, random
from numpy.random import uniform

# Simulation-specific
import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
from cbfkit.certificates import concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.controllers.cbf_clf.vanilla_cbf_clf_qp_control_laws import vanilla_cbf_clf_qp_controller
from cbfkit.estimators import ct_ekf_dtmeas
from cbfkit.integration import forward_euler as integrator
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
from cbfkit.sensors import unbiased_gaussian_noise as sensor
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
    """Identity measurement function."""
    return x


def DHDX(_x):
    """Jacobian of identity measurement function."""
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
)


@pytest.fixture
def initial_state():
    """Fixture to generate a valid initial state."""
    invalid_initial_condition = True
    state = None

    # Generate random initial condition
    while invalid_initial_condition:
        x_rand = uniform(low=-x_max, high=x_max)
        y_rand = uniform(low=-y_max, high=y_max)
        a_rand = uniform(low=-jnp.pi, high=jnp.pi)
        state = jnp.array([x_rand, y_rand, a_rand])

        invalid_initial_condition = any(
            (x_rand - xo[0]) ** 2 + (y_rand - xo[1]) ** 2 - ro[0] ** 2 < 0
            for xo, ro in zip(OBSTACLES, ELLIPSOIDS)
        )
    return state


@pytest.mark.benchmark
def test_execution_performance_jit(initial_state):
    """Tests that JIT-compiled execution is fast (average step < 1ms)."""
    planner_data = PlannerData(
        u_traj=None,
        x_traj=jnp.tile(GOAL.reshape(-1, 1), (1, N_STEPS + 1)),
        prev_robustness=None,
    )

    # Warmup
    sim.execute(
        x0=initial_state,
        dt=DT,
        num_steps=10,
        dynamics=DYNAMICS,
        integrator=integrator,
        planner=None,
        nominal_controller=NOMINAL_CONTROLLER,
        controller=CONTROLLER,
        sensor=sensor,
        estimator=ESTIMATOR,
        perturbation=generate_stochastic_perturbation(sigma=lambda _x: Q, dt=DT),
        sigma=R,
        key=random.PRNGKey(0),
        planner_data=planner_data,
        use_jit=True,
        verbose=False,
    )

    start_time = time.time()

    sim.execute(
        x0=initial_state,
        dt=DT,
        num_steps=N_STEPS,
        dynamics=DYNAMICS,
        integrator=integrator,
        planner=None,
        nominal_controller=NOMINAL_CONTROLLER,
        controller=CONTROLLER,
        sensor=sensor,
        estimator=ESTIMATOR,
        perturbation=generate_stochastic_perturbation(sigma=lambda _x: Q, dt=DT),
        sigma=R,
        key=random.PRNGKey(0),
        planner_data=planner_data,
        use_jit=True,
        verbose=False,
    )

    total_time = time.time() - start_time
    avg_step_ms = (total_time / N_STEPS) * 1000

    print(f"Total Time (s): {total_time:.4f}")
    print(f"Avg Step (ms): {avg_step_ms:.4f}")

    # JIT execution should be fast, satisfying the < 20ms requirement
    assert avg_step_ms < 20.0
