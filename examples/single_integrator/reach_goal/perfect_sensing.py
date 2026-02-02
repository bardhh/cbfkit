import pickle
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array, random

import cbfkit.simulation.simulator as sim
from cbfkit.controllers.cbf_clf.risk_aware_cbf_clf_qp_control_laws import (
    risk_aware_cbf_clf_qp_controller,
)
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.simulation.monte_carlo import conduct_monte_carlo
from cbfkit.systems import single_integrator
from examples.single_integrator.common.config import perfect_state_measurements as setup
from examples.single_integrator.common.lyapunov_functions import fxts_lyapunov

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


ra_clf_params = RiskAwareParams(
    t_max=setup.Tg,
    p_bound=setup.pg,
    gamma=setup.gamma_v,
    eta=setup.eta_v,
    sigma=lambda x: jnp.sqrt(setup.Q),
    varsigma=lambda x: jnp.sqrt(setup.R),
)

controller = risk_aware_cbf_clf_qp_controller(
    dynamics_func=dynamics,
    lyapunovs=lyapunovs,
    control_limits=setup.actuation_limits,
    alpha=jnp.array([0.1]),
    ra_clf_params=ra_clf_params,
)
# controller = ra_controllers.pi_cbf_clf_controller(
#     nominal_input=nominal_controller,
#     dynamics_func=dynamics,
#     lyapunovs=lyapunovs,
#     control_limits=setup.actuation_limits,
#     alpha=np.array([0.1]),
#     t_max=setup.Tg,
#     p_bound_v=setup.pg,
#     gamma_v=setup.gamma_v,
#     eta_v=setup.eta_v,
#     dt=setup.dt,
# )


def execute(
    ii: int = 0,
) -> Tuple[Array, Array, Array, Array, List[str], List[Array], List[str], List[Array], bool, float]:
    """_summary_

    Args:
        int (ii): _description_

    Returns
    -------
        List[Array]: _description_
    """
    key = random.PRNGKey(ii)
    k1, k2, k3 = random.split(key, 3)
    initial_quadrant = random.randint(k1, minval=1, maxval=5, shape=())
    initial_angle = float(
        np.pi / 2 * initial_quadrant + np.pi / 4 + (random.uniform(k2) - 0.5) * np.pi / 4
    )
    initial_radius = float(1.0 + random.uniform(k3) * (np.sqrt(2) - 1.0))
    initial_state = jnp.array(
        [initial_radius * jnp.cos(initial_angle), initial_radius * jnp.sin(initial_angle)]
    )

    states, controls, estimates, covariances, c_keys, c_values, p_keys, p_values = sim.execute(
        x0=initial_state,
        dynamics=dynamics,
        sensor=sensor,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=setup.dt,
        num_steps=N_STEPS,
    )

    success = jnp.linalg.norm(states[-1]) < setup.goal_radius
    completion_time = states.shape[0] * setup.dt
    print(f"Completed Trial No. {ii}")

    return (
        states,
        controls,
        estimates,
        covariances,
        c_keys,
        c_values,
        p_keys,
        p_values,
        success,
        completion_time,
    )


# Simulate total number of trials
if __name__ == "__main__":
    import os
    import pickle

    # Execute numerous trials sim
    results = conduct_monte_carlo(execute, n_trials=setup.n_trials)
    # results = [execute(0)]

    # Convert the results to a NumPy array
    states = [result[0] for result in results]
    controls = [result[1] for result in results]
    estimates = [result[2] for result in results]
    covariances = [result[3] for result in results]
    successes = [result[8] for result in results]
    completion_times = [result[9] if result[8] else None for result in results]

    n_success = jnp.sum(jnp.array(successes))
    fraction_success = n_success / setup.n_trials
    print(f"Success Rate: {fraction_success:.2f}")

    valid_completion_times = [t for t in completion_times if t is not None]
    if valid_completion_times:
        print(f"Avg. Completion Time: {jnp.mean(jnp.array(valid_completion_times)):.2f}")
    else:
        print("Avg. Completion Time: N/A")

    # states, controls, estimates, covariances, data_keys, data_values, successes = execute()

    save_data = {
        "x": states,
        "u": controls,
        "z": estimates,
        "p": covariances,
        "pg": setup.pg,
        "successes": successes,
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(setup.pkl_file), exist_ok=True)

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
                animation_filename="examples/single_integrator/reach_goal/results/perfect_measurements",
            )
