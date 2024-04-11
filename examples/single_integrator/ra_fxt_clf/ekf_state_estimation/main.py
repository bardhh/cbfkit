"""
This module simulates a 6 degree-of-freedom dynamic quadrotor model as it seeks
to reach a goal region while avoiding dynamic obstacles.

"""

from typing import List
from jax import Array, random, jacfwd
import numpy as np

# import jax.numpy as jnp
import cbfkit.system as system
from cbfkit.monte_carlo import conduct_monte_carlo
from cbfkit.models import single_integrator
from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.estimators import ct_ekf_dtmeas
from cbfkit.integration import forward_euler as integrator

# import cbfkit.controllers.model_based.cbf_clf_controllers.risk_aware_cbf_clf_controllers as ra_controllers
import cbfkit.controllers.model_based.cbf_clf_controllers.estimated_risk_aware_cbf_clf_controllers as est_ra_controllers
from examples.single_integrator.ra_fxt_clf.ekf_state_estimation import setup

from examples.single_integrator.ra_fxt_clf.lyapunov_functions import fxts_lyapunov

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
dynamics = single_integrator.two_dimensional_single_integrator(r=setup.goal_radius, sigma=setup.Q)
nominal_controller = single_integrator.zero_controller()
dfdx = lambda x: jacfwd(dynamics)(x)
h = lambda x: x
dhdx = lambda _x: np.eye((len(setup.desired_state)))
estimator = ct_ekf_dtmeas(
    Q=setup.Q,
    R=setup.R,
    dynamics=dynamics,
    dfdx=dfdx,
    h=h,
    dhdx=dhdx,
    dt=setup.dt,
)


# controller = ra_controllers.fxt_cbf_clf_controller(
#     nominal_input=nominal_controller,
#     dynamics_func=dynamics,
#     lyapunovs=lyapunovs,
#     control_limits=setup.actuation_limits,
#     alpha=np.array([0.1]),
#     t_max=setup.Tg,
#     p_bound_v=setup.pg,
#     gamma_v=setup.gamma_v,
#     eta_v=setup.eta_v,
# )
# controller = est_ra_controllers.fxt_cbf_clf_controller(
#     nominal_input=nominal_controller,
#     dynamics_func=dynamics,
#     lyapunovs=lyapunovs,
#     control_limits=setup.actuation_limits,
#     alpha=np.array([0.1]),
#     t_max=setup.Tg,
#     p_bound_v=setup.pg,
#     gamma_v=setup.gamma_v,
#     eta_v=setup.eta_v,
#     varsigma=setup.R,
# )
controller = est_ra_controllers.pi_cbf_clf_controller(
    nominal_input=nominal_controller,
    dynamics_func=dynamics,
    lyapunovs=lyapunovs,
    control_limits=setup.actuation_limits,
    alpha=np.array([0.1]),
    t_max=setup.Tg,
    p_bound_v=setup.pg,
    gamma_v=setup.gamma_v,
    eta_v=setup.eta_v,
    varsigma=setup.R,
    dt=setup.dt,
)


def execute(ii: int = 0) -> List[Array]:
    """_summary_

    Args:
        int (ii): _description_

    Returns:
        List[Array]: _description_
    """
    key = random.PRNGKey(ii)
    initial_quadrant = random.randint(key, minval=1, maxval=5, shape=(1,))
    initial_angle = float(
        np.pi / 2 * initial_quadrant + np.pi / 4 + (random.uniform(key) - 0.5) * np.pi / 4
    )
    initial_radius = float(1.0 + random.uniform(key) * (np.sqrt(2) - 1.0))
    initial_state = np.array(
        [initial_radius * np.cos(initial_angle), initial_radius * np.sin(initial_angle)]
    )

    x, u, z, p, dkeys, dvalues = system.execute(
        x0=initial_state,
        dynamics=dynamics,
        sensor=sensor,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=setup.dt,
        R=setup.R,
        num_steps=N_STEPS,
        key=key,
    )

    # Reformat results as numpy arrays
    x = np.array(x)
    u = np.array(u)
    z = np.array(z)
    p = np.array(p)
    success = np.linalg.norm(x[-1]) < setup.goal_radius
    completion_time = x.shape[0] * setup.dt
    print(f"Completed Trial No. {ii}")

    return x, u, z, p, dkeys, dvalues, success, completion_time


# Simulate total number of trials
if __name__ == "__main__":
    import pickle

    # Execute numerous trials sim
    results = conduct_monte_carlo(execute, n_trials=setup.n_trials)

    # Convert the results to a NumPy array
    states = [result[0] for result in results]
    controls = [result[1] for result in results]
    estimates = [result[2] for result in results]
    covariances = [result[3] for result in results]
    successes = [result[6] for result in results]
    completion_times = [result[7] for result in results if result[6]]

    n_success = np.sum(successes)
    fraction_success = n_success / setup.n_trials
    print(f"Success Rate: {fraction_success:.2f}")
    print(f"Avg. Completion Time: {np.mean(completion_times):.2f}")

    # (
    #     states,
    #     controls,
    #     estimates,
    #     covariances,
    #     data_keys,
    #     data_values,
    #     successes,
    #     completion_times,
    # ) = execute()

    save_data = {
        "x": states,
        "u": controls,
        "z": estimates,
        "p": covariances,
        "pg": setup.pg,
        "successes": successes,
    }

    # Save data to file
    with open(setup.pkl_file, "wb") as file:
        pickle.dump(save_data, file)

    # Visualizations
    if setup.VISUALIZE:
        from examples.van_der_pol.visualizations.path import animate

        for trial_no in range(setup.n_trials):
            animate(
                states=states[trial_no],
                estimates=estimates[trial_no],
                desired_state=setup.desired_state,
                desired_state_radius=setup.goal_radius,
                x_lim=(-2, 2),
                y_lim=(-2, 2),
                dt=setup.dt,
                title="System Behavior",
                save_animation=False,
                animation_filename="examples/single_integrator/ra_fxt_clf/ekf_state_estimation/results/test",
            )
