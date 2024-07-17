"""
This module simulates a 6 degree-of-freedom dynamic quadrotor model as it seeks
to reach a goal region while avoiding dynamic obstacles.

"""

import numpy as np
import cbfkit.simulation.simulator as sim
from cbfkit.systems.fixed_wing_uav.models import beard2014_kinematic as fixed_wing_uav
from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.estimators import ct_ekf_dtmeas
from cbfkit.integration import forward_euler as integrator
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.risk_aware_cbf_clf_qp_control_laws import (
    estimate_feedback_risk_aware_cbf_clf_qp_controller,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.risk_aware_params import (
    RiskAwareParams,
)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.lyapunov_conditions.fixed_time_stability import (
    fxt_s,
)

from examples.fixed_wing.reach_drop_point import setup

lyapunov_barrier_package = (
    fixed_wing_uav.certificate_functions.barrier_lyapunov_functions.velocity_with_obstacles
)
lyapunov_barriers = lyapunov_barrier_package(
    certificate_conditions=fxt_s(c1=setup.c1, c2=setup.c2, e1=setup.e1, e2=setup.e2),
    goal=setup.desired_velpos,
    rg=setup.goal_radius,
    obstacles=setup.obstacle_locations,
    robs=setup.ellipsoid_radii,
    alpha=setup.alpha,
)

# number of simulation steps
N_STEPS = int(setup.tf / setup.dt)

# Define dynamics, controller, and estimator with specified parameters
dynamics = fixed_wing_uav.plant()
dfdx = fixed_wing_uav.plant_jacobians()

nominal_controller = fixed_wing_uav.controllers.zero_controller()
h = lambda x: x
dhdx = lambda _x: np.eye((len(setup.initial_state)))
estimator = ct_ekf_dtmeas(
    Q=setup.Q,
    R=setup.R,
    dynamics=dynamics,
    dfdx=dfdx,
    h=h,
    dhdx=dhdx,
    dt=setup.dt,
)

risk_aware_clf_params = RiskAwareParams(
    t_max=setup.Tg,
    p_bound=setup.pg,
    gamma=setup.gamma_v,
    eta=setup.eta_v,
    epsilon=setup.epsilon,
    lambda_h=setup.lambda_h,
    lambda_generator=setup.lambda_generator,
    varsigma=setup.R,
)

controller = estimate_feedback_risk_aware_cbf_clf_qp_controller(
    nominal_input=nominal_controller,
    dynamics_func=dynamics,
    lyapunovs=lyapunov_barriers,
    control_limits=setup.actuation_limits,
    R=setup.QP_J,
    ra_clf_params=risk_aware_clf_params,
)


x, u, z, p, dkeys, dvalues = sim.execute(
    x0=setup.initial_state,
    dt=setup.dt,
    num_steps=N_STEPS,
    dynamics=dynamics,
    integrator=integrator,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    sigma=setup.R,
    perturbation=generate_stochastic_perturbation(sigma=lambda x: setup.Q, dt=setup.dt),
)

save_data = {
    "x": x,
    "u": u,
    "z": z,
    "p": p,
    "pg": setup.pg,
}

# Save data to file
import pickle

with open(setup.pkl_file, "wb") as file:
    pickle.dump(save_data, file)


from examples.fixed_wing.reach_drop_point.visualizations.animate_2d_path import (
    animate as animate_2d,
)

from examples.fixed_wing.reach_drop_point.visualizations.path_3d import animate as animate_3d

animate_3d(
    trajectory=x,
    obstacles=setup.obstacle_locations,
    r_obs=setup.ellipsoid_radii,
    dt=setup.dt,
    save_animation=True,
    animation_filename="examples/fixed_wing/reach_drop_point/results/3d_animation",
)
animate_2d(
    states=x,
    estimates=z,
    desired_state=setup.desired_state,
    desired_state_radius=0.1,
    obstacles=setup.obstacle_locations,
    r_obs=setup.ellipsoid_radii,
    x_lim=(-100, 1000),
    y_lim=(-100, 500),
    dt=setup.dt,
    title="System Behavior",
    save_animation=True,
    animation_filename="examples/fixed_wing/reach_drop_point/results/2d_animation",
)
