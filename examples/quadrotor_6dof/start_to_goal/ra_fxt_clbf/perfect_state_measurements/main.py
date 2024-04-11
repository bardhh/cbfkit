"""
This module simulates a 6 degree-of-freedom dynamic quadrotor model as it seeks
to reach a goal region while avoiding dynamic obstacles.

"""

import numpy as np

# Whether or not to simulate, plot
simulate = 1
plot = 1
save = 0

# Load system module
import cbfkit.system as system

# Load dynamics, sensors, controller, estimator, integrator
from cbfkit.models import quadrotor_6dof
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator
from cbfkit.controllers.model_based.cbf_clf_controllers.cbf_clf_qp_controllers import (
    cbf_clf_controller,
)

# Load initial conditions
from examples.quadrotor_6dof.start_to_goal.ra_fxt_clbf.perfect_state_measurements import (
    initial_conditions,
)
from examples.quadrotor_6dof.start_to_goal.ra_fxt_clbf.barrier_functions import barriers
from examples.quadrotor_6dof.start_to_goal.ra_fxt_clbf.lyapunov_functions import (
    fxts_lyapunovs,
    fxts_geometric_lyapunovs,
)

# lyapunovs = fxts_geometric_lyapunovs(
#     initial_conditions.desired_state,
#     initial_conditions.m,
#     initial_conditions.c1,
#     initial_conditions.c2,
#     initial_conditions.e1,
#     initial_conditions.e2,
# )

# number of simulation steps
n_steps = int(initial_conditions.tf / initial_conditions.dt)

# Define dynamics, controller, and estimator with specified parameters
dynamics = quadrotor_6dof.quadrotor_6dof_dynamics(
    m=initial_conditions.m,
    jx=initial_conditions.jx,
    jy=initial_conditions.jy,
    jz=initial_conditions.jz,
)
controller = quadrotor_6dof.geometric_controller(
    dynamics=dynamics,
    desired_state=initial_conditions.desired_state,
    dt=initial_conditions.dt,
    m=initial_conditions.m,
    jx=initial_conditions.jx,
    jy=initial_conditions.jy,
    jz=initial_conditions.jz,
)
# nominal_controller = quadrotor_6dof.zero_controller()

# controller = cbf_clf_controller(
#     nominal_input=nominal_controller,
#     dynamics_func=dynamics,
#     # barriers=barriers,
#     lyapunovs=lyapunovs,
#     control_limits=initial_conditions.actuation_limits,
#     alpha=np.array([0.1]),
# )


if simulate:
    # Execute simulation
    states, controls, estimates, covariances, data_keys, data_values = system.execute(
        x0=initial_conditions.initial_state,
        dynamics=dynamics,
        sensor=sensor,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=initial_conditions.dt,
        R=initial_conditions.R,
        num_steps=n_steps,
    )

    # Reformat results as numpy arrays
    states = np.array(states)
    controls = np.array(controls)
    estimates = np.array(estimates)
    covariances = np.array(covariances)

else:
    # Implement load from file
    pass

if plot:
    from examples.quadrotor_6dof.start_to_goal.visualizations import animate

    animate(
        states=states,
        estimates=estimates,
        desired_state=initial_conditions.desired_state,
        desired_state_radius=0.1,
        x_lim=(-1, 11),
        y_lim=(-1, 11),
        dt=initial_conditions.dt,
        title="System Behavior",
        save_animation=save,
        animation_filename="examples/quadrotor_6dof/start_to_goal/ra_fxt_clbf/perfect_state_measurements/results/test",
    )
