"""
This module simulates a 6 degree-of-freedom dynamic quadrotor model as it seeks
to reach a goal region while avoiding dynamic obstacles.

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

import matplotlib

# Hack to prevent matplotlib.use("macosx") error in imported modules
matplotlib.use = lambda *args, **kwargs: None

from typing import List

import numpy as np
from jax import Array
import jax.numpy as jnp

import cbfkit.simulation.simulator as system
from cbfkit.controllers.cbf_clf.risk_aware_cbf_clf_qp_control_laws import (
    risk_aware_cbf_clf_qp_controller as cbf_clf_controller,
)
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.systems import van_der_pol
from cbfkit.sensors import perfect as sensor
from examples.van_der_pol.common.lyapunov_functions import fxts_lyapunov
from examples.van_der_pol.common.config import perfect_state_measurements as setup

# Lyapunov Barrier Functions
lyapunovs = fxts_lyapunov(
    setup.desired_state,
    setup.goal_radius,
    setup.c1,
    setup.c2,
    setup.e1,
    setup.e2,
)()

# number of simulation steps
N_STEPS = int(setup.tf / setup.dt)

# Define dynamics, controller, and estimator with specified parameters
dynamics = van_der_pol.reverse_van_der_pol_oscillator(epsilon=setup.epsilon, sigma=setup.Q)

risk_aware_clf_params = RiskAwareParams(
    t_max=setup.Tg,
    p_bound=setup.pg,
    gamma=setup.gamma_v,
    eta=setup.eta_v,
    epsilon=setup.epsilon,
    lambda_h=setup.lambda_h,
    lambda_generator=setup.lambda_generator,
    sigma=lambda x: setup.Q,
)

nominal_controller = van_der_pol.zero_controller()
controller = cbf_clf_controller(
    nominal_input=nominal_controller,
    dynamics_func=dynamics,
    lyapunovs=lyapunovs,
    control_limits=setup.actuation_limits,
    ra_clf_params=risk_aware_clf_params,
)


def execute(_ii: int = 1) -> List[Array]:
    """_summary_

    Args:
        int (ii): _description_

    Returns:
        List[Array]: _description_
    """
    (
        x,
        u,
        z,
        p,
        dkeys,
        dvalues,
        planner_keys,
        planner_values,
    ) = system.execute(
        x0=setup.initial_state,
        dynamics=dynamics,
        sensor=sensor,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=setup.dt,
        sigma=setup.R,
        num_steps=N_STEPS,
    )

    # Reformat results as numpy arrays
    x = np.array(x)
    u = np.array(u)
    z = np.array(z)
    p = np.array(p)

    return x, u, z, p, dkeys, dvalues


states, controls, estimates, covariances, data_keys, data_values = execute()

from examples.van_der_pol.visualizations.path import animate

animate(
    states=states,
    estimates=estimates,
    desired_state=setup.desired_state,
    desired_state_radius=setup.goal_radius,
    x_lim=(-abs(setup.initial_state[0]), abs(setup.initial_state[0])),
    y_lim=(-abs(setup.initial_state[0]), abs(setup.initial_state[0])),
    dt=setup.dt,
    title="System Behavior",
    save_animation=False,
    animation_filename="examples/van_der_pol/ra_fxt_clbf/perfect_state_measurements/results/test",
)
