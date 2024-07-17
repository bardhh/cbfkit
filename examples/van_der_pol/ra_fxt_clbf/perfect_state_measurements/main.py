"""
This module simulates a 6 degree-of-freedom dynamic quadrotor model as it seeks
to reach a goal region while avoiding dynamic obstacles.

"""

from typing import List
from jax import Array
import numpy as np
import cbfkit.system as system
from cbfkit.models import van_der_pol
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.risk_aware_cbf_clf_controllers import (
    cbf_clf_controller,
)
from examples.van_der_pol.ra_fxt_clbf.perfect_state_measurements import setup

from examples.van_der_pol.ra_fxt_clbf.lyapunov_functions import fxts_lyapunov

# Lyapunov Barrier Functions
lyapunovs = fxts_lyapunov(
    setup.desired_state,
    setup.goal_radius,
    setup.c1,
    setup.c2,
    setup.e1,
    setup.e2,
)

# number of simulation steps
N_STEPS = int(setup.tf / setup.dt)

# Define dynamics, controller, and estimator with specified parameters
dynamics = van_der_pol.reverse_van_der_pol_oscillator(epsilon=setup.epsilon, sigma=setup.Q)
# controller = van_der_pol.fxt_lyapunov_controller(epsilon=setup.epsilon)
# controller = van_der_pol.fxt_stochastic_lyapunov_controller(
#     epsilon=setup.epsilon,
#     sigma_sum_squares=np.sum(setup.Q**2),
# )
# controller = van_der_pol.fxt_risk_aware_lyapunov_controller(
#     epsilon=setup.epsilon,
#     sigma_sum_squares=np.sum(setup.Q**2),
#     pg=setup.pg,
#     t_reach=setup.Tg,
#     vartheta=2 * np.dot(np.array([2.0, 2.0]), np.diagonal(setup.Q)),
# )

nominal_controller = van_der_pol.zero_controller()
controller = cbf_clf_controller(
    nominal_input=nominal_controller,
    dynamics_func=dynamics,
    lyapunovs=lyapunovs,
    control_limits=setup.actuation_limits,
    t_max=setup.Tg,
    p_bound_v=setup.pg,
    gamma_v=setup.gamma_v,
    eta_v=setup.eta_v,
)


def execute(_ii: int = 1) -> List[Array]:
    """_summary_

    Args:
        int (ii): _description_

    Returns:
        List[Array]: _description_
    """
    x, u, z, p, dkeys, dvalues = system.execute(
        x0=setup.initial_state,
        dynamics=dynamics,
        sensor=sensor,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=setup.dt,
        R=setup.R,
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
