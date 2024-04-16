"""
Test Module for cbfkit.controllers.model_based.cbf_clf_controllers control laws.
=========================

This module contains unit tests for functionalities in 'cbf_clf_controllers'
from 'cbfkit.controllers.model_based'.

Tests
-----
The following test that the given controller is set up correctly:
- test_setup_risk_aware_cbf_clf_qp_control_law
- test_setup_path_integral_risk_aware_cbf_clf_qp_control_law
- test_setup_robust_cbf_clf_qp_control_law
- test_setup_stochastic_cbf_clf_qp_control_law
- test_setup_vanilla_cbf_clf_qp_control_law

Setup
-----
- No set up required

Examples
--------
To run all tests in this module (from the root of the repository):
    $ python -m unittest tests.test_controllers.test_cbf_clf_controllers.test_cbf_clf_qp_controllers
"""

import unittest
import jax.numpy as jnp
from jax import random
import cbfkit.controllers.model_based.cbf_clf_controllers as cbf_clf_controllers
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.risk_aware_params import (
    RiskAwareParams,
)

KEY = random.PRNGKey(0)

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
GOAL = jnp.array([0.0, 0.0, 0.0])
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

DYNAMICS = unicycle.plant(l=1.0)
NOMINAL_CONTROLLER = unicycle.controllers.proportional_controller(
    dynamics=DYNAMICS,
    Kp_pos=1.0,
    Kp_theta=0.01,
    desired_state=GOAL,
)
RISK_AWARE_CBF_PARAMS = RiskAwareParams(
    t_max=5.0,
    p_bound=0.25,
    gamma=0.5,
    eta=10.0,
    sigma=lambda x: Q,
)
RISK_AWARE_CLF_PARAMS = RiskAwareParams(
    t_max=5.0,
    p_bound=0.75,
    gamma=10.0,
    eta=10.0,
    sigma=lambda x: Q,
)


class TestCbfClfQpControllers(unittest.TestCase):
    """Takes care of unit tests intended to verify the intended performance
    of CBF-CLF-QP control laws.

    """

    eps = 1e-2

    def test_setup_risk_aware_cbf_clf_qp_control_law(self):
        """Tests that the Risk-Aware (RA)-CBF-CLF-QP controller is set up correctly."""
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions.risk_aware_barrier import (
            right_hand_side,
        )
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
            e_s,
        )

        # Barrier functions
        bars = [
            unicycle.certificate_functions.barrier_functions.obstacle_ca(
                certificate_conditions=right_hand_side(RISK_AWARE_CBF_PARAMS.p_bound, ALPHA),
                obstacle=jnp.array([obs[0], obs[1], 0.0]),
                ellipsoid=jnp.array([ell[0], ell[1]]),
            )
            for obs, ell in zip(OBSTACLES, ELLIPSOIDS)
        ]
        BARRIERS = concatenate_certificates(*bars)

        # Lyapunov functions
        lyaps = unicycle.certificate_functions.lyapunov_functions.reach_goal(
            certificate_conditions=e_s(c=2.0),
            goal=GOAL,
            radius=RAD,
        )

        LYAPUNOVS = concatenate_certificates(lyaps)

        # Create controller
        controller = cbf_clf_controllers.risk_aware_cbf_clf_qp_controller(
            control_limits=ACTUATION_LIMITS,
            nominal_input=NOMINAL_CONTROLLER,
            dynamics_func=DYNAMICS,
            barriers=BARRIERS,
            lyapunovs=LYAPUNOVS,
            ra_cbf_params=RISK_AWARE_CBF_PARAMS,
            ra_clf_params=RISK_AWARE_CLF_PARAMS,
        )

        u, _ = controller(0.0, jnp.zeros((N,)))

        self.assertTrue(u.shape == (M,))

    def test_setup_risk_aware_path_integral_cbf_clf_qp_control_law(self):
        """Tests that the Risk-Aware Path Integral (RA-PI)-CBF-CLF-QP controller is set up correctly."""
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions.path_integral_barrier import (
            right_hand_side,
        )
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
            e_s,
        )

        # Barrier functions
        bars = [
            unicycle.certificate_functions.barrier_functions.obstacle_ca(
                certificate_conditions=right_hand_side(
                    RISK_AWARE_CBF_PARAMS.p_bound,
                    RISK_AWARE_CBF_PARAMS.gamma,
                    RISK_AWARE_CBF_PARAMS.eta,
                    RISK_AWARE_CBF_PARAMS.t_max,
                ),
                obstacle=jnp.array([obs[0], obs[1], 0.0]),
                ellipsoid=jnp.array([ell[0], ell[1]]),
            )
            for obs, ell in zip(OBSTACLES, ELLIPSOIDS)
        ]
        BARRIERS = concatenate_certificates(*bars)

        # Lyapunov functions
        lyaps = unicycle.certificate_functions.lyapunov_functions.reach_goal(
            certificate_conditions=e_s(c=2.0),
            goal=GOAL,
            radius=RAD,
        )

        LYAPUNOVS = concatenate_certificates(lyaps)

        # Create controller
        controller = cbf_clf_controllers.risk_aware_path_integral_cbf_clf_qp_controller(
            control_limits=ACTUATION_LIMITS,
            nominal_input=NOMINAL_CONTROLLER,
            dynamics_func=DYNAMICS,
            barriers=BARRIERS,
            lyapunovs=LYAPUNOVS,
            ra_cbf_params=RISK_AWARE_CBF_PARAMS,
            ra_clf_params=RISK_AWARE_CLF_PARAMS,
        )

        u, _ = controller(0.0, jnp.zeros((N,)))

        self.assertTrue(u.shape == (M,))

    def test_setup_robust_cbf_clf_qp_control_law(self):
        """Tests that the Robust (R)-CBF-CLF-QP controller is set up correctly."""
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions.zeroing_barriers import (
            linear_class_k,
        )
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
            e_s,
        )

        # Barrier functions
        bars = [
            unicycle.certificate_functions.barrier_functions.obstacle_ca(
                certificate_conditions=linear_class_k(alpha=1.0),
                obstacle=jnp.array([obs[0], obs[1], 0.0]),
                ellipsoid=jnp.array([ell[0], ell[1]]),
            )
            for obs, ell in zip(OBSTACLES, ELLIPSOIDS)
        ]
        BARRIERS = concatenate_certificates(*bars)

        # Lyapunov functions
        lyaps = unicycle.certificate_functions.lyapunov_functions.reach_goal(
            certificate_conditions=e_s(c=2.0),
            goal=GOAL,
            radius=RAD,
        )

        LYAPUNOVS = concatenate_certificates(lyaps)

        # Create controller
        controller = cbf_clf_controllers.robust_cbf_clf_qp_controller(
            control_limits=ACTUATION_LIMITS,
            nominal_input=NOMINAL_CONTROLLER,
            dynamics_func=DYNAMICS,
            barriers=BARRIERS,
            lyapunovs=LYAPUNOVS,
            disturbance_norm=2,
            disturbance_norm_bound=10.0,
        )

        u, _ = controller(0.0, jnp.zeros((N,)))

        self.assertTrue(u.shape == (M,))

    def test_setup_stochastic_cbf_clf_qp_control_law(self):
        """Tests that the Stochastic (S)-CBF-CLF-QP controller is set up correctly."""
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions.stochastic_barrier import (
            right_hand_side,
        )
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
            e_s,
        )

        # Barrier functions
        bars = [
            unicycle.certificate_functions.barrier_functions.obstacle_ca(
                certificate_conditions=right_hand_side(
                    alpha=1.0,
                    beta=0.25,
                ),
                obstacle=jnp.array([obs[0], obs[1], 0.0]),
                ellipsoid=jnp.array([ell[0], ell[1]]),
            )
            for obs, ell in zip(OBSTACLES, ELLIPSOIDS)
        ]
        BARRIERS = concatenate_certificates(*bars)

        # Lyapunov functions
        lyaps = unicycle.certificate_functions.lyapunov_functions.reach_goal(
            certificate_conditions=e_s(c=2.0),
            goal=GOAL,
            radius=RAD,
        )

        LYAPUNOVS = concatenate_certificates(lyaps)

        # Create controller
        controller = cbf_clf_controllers.stochastic_cbf_clf_qp_controller(
            control_limits=ACTUATION_LIMITS,
            nominal_input=NOMINAL_CONTROLLER,
            dynamics_func=DYNAMICS,
            barriers=BARRIERS,
            lyapunovs=LYAPUNOVS,
            sigma=lambda x: Q,
        )

        u, _ = controller(0.0, jnp.zeros((N,)))

        self.assertTrue(u.shape == (M,))

    def test_setup_vanilla_cbf_clf_qp_control_law(self):
        """Tests that the vanilla CBF-CLF-QP controller is set up correctly."""
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions.zeroing_barriers import (
            linear_class_k,
        )
        from cbfkit.controllers.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
            e_s,
        )

        # Barrier functions
        bars = [
            unicycle.certificate_functions.barrier_functions.obstacle_ca(
                certificate_conditions=linear_class_k(alpha=1.0),
                obstacle=jnp.array([obs[0], obs[1], 0.0]),
                ellipsoid=jnp.array([ell[0], ell[1]]),
            )
            for obs, ell in zip(OBSTACLES, ELLIPSOIDS)
        ]
        BARRIERS = concatenate_certificates(*bars)

        # Lyapunov functions
        lyaps = unicycle.certificate_functions.lyapunov_functions.reach_goal(
            certificate_conditions=e_s(c=2.0),
            goal=GOAL,
            radius=RAD,
        )

        LYAPUNOVS = concatenate_certificates(lyaps)

        # Create controller
        controller = cbf_clf_controllers.vanilla_cbf_clf_qp_controller(
            control_limits=ACTUATION_LIMITS,
            nominal_input=NOMINAL_CONTROLLER,
            dynamics_func=DYNAMICS,
            barriers=BARRIERS,
            lyapunovs=LYAPUNOVS,
        )

        u, _ = controller(0.0, jnp.zeros((N,)))

        self.assertTrue(u.shape == (M,))
