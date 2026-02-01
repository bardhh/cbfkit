"""
Test Module for simulating the fixed-wing UAV system
=========================

This module contains tests for simulating the fixed-wing UAV system under
various model dynamics.

Tests
-----
Reach the drop point:
- test_fixed_wing_risk_aware_fxt_clf_controller
- test_fixed_wing_clf_controller

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

import cbfkit.simulation.simulator as sim
from cbfkit.controllers.model_based.cbf_clf_controllers import vanilla_cbf_clf_qp_controller
from cbfkit.controllers.model_based.cbf_clf_controllers.risk_aware_cbf_clf_qp_control_laws import (
    risk_aware_cbf_clf_qp_controller,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.risk_aware_params import (
    RiskAwareParams,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.lyapunov_conditions.fixed_time_stability import (
    fxt_s,
)
from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.estimators import ct_ekf_dtmeas
from cbfkit.integration import forward_euler as integrator
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation


# Simulation-specific
import cbfkit.systems.fixed_wing_uav.models.beard2014_kinematic as uav

# Simulation setup
N = 6  # n_states
M = 3  # n_controls
TF = 1.0
DT = 1e-2
N_STEPS = int(TF / DT)
INITIAL_STATE = jnp.array([500.0, 250.0, 250.0, 100.0, -jnp.pi / 2, 0.0])


# Control params
ACTUATION_LIMITS = jnp.array([1e3, 1e3, jnp.tan(jnp.pi / 2.01)])

# Stochastic noise parameters
Q = 0.05 * jnp.eye(N)  # process noise
R = 0.01 * jnp.eye(N)  # measurement noise
PG = 0.5
GAMMA_V = 100.0
ETA_V = float(jnp.linalg.norm(jnp.dot(10.0 * jnp.ones((N,)), Q)))

# Lyapunov function parameters
C1 = 4.0
C2 = 4.0
E1 = 0.9
E2 = 1.1
GOAL = jnp.array([-100.0, 0.0, 200.0])
RAD = 10.0

# Lyapunov function
l1 = uav.certificate_functions.lyapunov_functions.velocity(
    certificate_conditions=fxt_s(C1, C2, E1, E2),
    goal=GOAL,
    r=RAD,
)
LYAPUNOVS = concatenate_certificates(l1)

DYNAMICS = uav.plant()
NOMINAL_CONTROLLER = uav.controllers.zero_controller()
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


def execute(controller) -> List[Array]:
    """_summary_

    Args:
        int (ii): _description_

    Returns:
        List[Array]: _description_
    """
    x, _u, _z, _p, dkeys, dvalues = sim.execute(
        x0=INITIAL_STATE,
        dt=DT,
        num_steps=N_STEPS,
        dynamics=DYNAMICS,
        perturbation=generate_stochastic_perturbation(sigma=lambda _x: Q, dt=DT),
        integrator=integrator,
        controller=controller,
        sensor=sensor,
        estimator=ESTIMATOR,
        sigma=R,
    )

    if "error" in dkeys:
        if dvalues[-1][dkeys.index("error")]:
            return False, x.shape[0], x[-1]

    return True, x.shape[0], x[-1]


class TestFixedWingSimulation(unittest.TestCase):
    """Takes care of unit tests intended to simulate versions of the
    fixed-wing UAV reach drop point case study.

    """

    def test_fixed_wing_risk_aware_fxt_clf_controller(self):
        """Tests the fixed-wing UAV simulation with a RA-FxT-CLF controller."""
        ra_clf_params = RiskAwareParams(
            t_max=1 / (C1 * (1 - E1)) + 1 / (C2 * (E2 - 1)),
            p_bound=PG,
            gamma=GAMMA_V,
            eta=ETA_V,
            sigma=lambda _: Q,
            varsigma=lambda _: R,
        )
        controller = risk_aware_cbf_clf_qp_controller(
            nominal_input=NOMINAL_CONTROLLER,
            dynamics_func=DYNAMICS,
            lyapunovs=LYAPUNOVS,
            control_limits=ACTUATION_LIMITS,
            ra_clf_params=ra_clf_params,
            relaxable_clf=True,
        )
        success, n_iter, xf = execute(controller)
        self.assertTrue(
            success, msg=f"Completed {n_iter}/{N_STEPS} iterations\nTerminal State: {xf}"
        )

    def test_fixed_wing_clf_controller(self):
        """Tests the fixed-wing UAV simulation with a CLF controller."""
        controller = vanilla_cbf_clf_qp_controller(
            nominal_input=NOMINAL_CONTROLLER,
            dynamics_func=DYNAMICS,
            lyapunovs=LYAPUNOVS,
            control_limits=ACTUATION_LIMITS,
            relaxable_clf=True,
        )
        success, n_iter, xf = execute(controller)
        self.assertTrue(
            success, msg=f"Completed {n_iter}/{N_STEPS} iterations\nTerminal State: {xf}"
        )


if __name__ == "__main__":
    unittest.main()
