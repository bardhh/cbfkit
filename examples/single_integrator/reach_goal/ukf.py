"""Single integrator reach-goal with UKF estimation and Monte Carlo evaluation."""
import sys
import os

# Add the project root to the path so we can import examples
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array, random
import cbfkit.simulation.simulator as sim
from cbfkit.certificates import concatenate_certificates
from cbfkit.controllers.cbf_clf.risk_aware_cbf_clf_qp_control_laws import (
    risk_aware_cbf_clf_qp_controller,
)
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.estimators import ct_ukf_dtmeas
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.simulation.monte_carlo import conduct_monte_carlo
from cbfkit.systems import single_integrator
from cbfkit.systems.single_integrator.certificates.lyapunov_function_catalog import position
from cbfkit.utils.user_types import CertificateCollection
from examples.single_integrator.common.config import ukf_state_estimation as setup
from examples.single_integrator.common.lyapunov_functions import fxts_lyapunov_conditions


# Local definition of fxts_lyapunov to return CertificateCollection
def fxts_lyapunov(
    goal: Array, r: float, c1: float, c2: float, e1: float, e2: float
) -> CertificateCollection:
    """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

    Args:
        goal (Array): goal location
        r (float): goal set radius
        c1 (float): convergence constant 1
        c2 (float): convergence constant 2
        e1 (float): exponential constant 1
        e2 (float): exponential constant 2

    Returns
    -------
        CertificateCollection: all inforrmation needed for CLF constraint in QP
    """
    pos_data = position(goal, r)
    cond_data = fxts_lyapunov_conditions(c1, c2, e1, e2)

    return CertificateCollection(
        functions=pos_data[0],
        jacobians=pos_data[1],
        hessians=pos_data[2],
        partials=pos_data[3],
        conditions=cond_data[0],
    )


# Local definition of animate to avoid backend issues
def animate(
    states,
    estimates,
    desired_state,
    desired_state_radius,
    x_lim=(-4, 4),
    y_lim=(-4, 4),
    dt=0.1,
    title="System Behavior",
    save_animation=False,
    animation_filename="system_behavior.gif",
):
    from cbfkit.utils.animator import CBFAnimator

    animator = CBFAnimator(states, dt=dt, x_lim=x_lim, y_lim=y_lim, title=title)
    animator.add_goal(desired_state[:2], radius=desired_state_radius)
    animator.add_trajectory(
        x_idx=0,
        y_idx=1,
        data=estimates,
        color="tab:orange",
        label="Estimated Trajectory",
        style="scatter",
        zorder=2,
    )
    animator.add_trajectory(
        x_idx=0,
        y_idx=1,
        color="tab:blue",
        label="Trajectory",
        zorder=3,
    )

    if save_animation:
        animator.save(animation_filename)
        print(f"  Animation saved: {animation_filename}")

    return animator.fig, animator.ax


# Lyapunov Barrier Functions
lyapunovs = concatenate_certificates(
    fxts_lyapunov(
        setup.desired_state,
        setup.goal_radius,
        setup.c1,
        setup.c2,
        setup.e1,
        setup.e2,
    )
)

# number of simulation steps
N_STEPS = int(setup.tf / setup.dt)

# Define dynamics, controller, and estimator with specified parameters
dynamics = single_integrator.two_dimensional_single_integrator(r=setup.goal_radius, sigma=setup.Q)
nominal_controller = single_integrator.zero_controller()


def h(x):
    return x


estimator = ct_ukf_dtmeas(
    Q=setup.Q,
    R=setup.R,
    dynamics=dynamics,
    h=h,
    dt=setup.dt,
)


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
    alpha=np.array([0.1]),
    ra_clf_params=ra_clf_params,
)


def execute(ii: int = 0) -> Tuple[Array, Array, Array, Array, List[str], List[Array], bool, float]:
    key = random.PRNGKey(ii)
    k1, k2, k3 = random.split(key, 3)
    initial_quadrant = random.randint(k1, minval=1, maxval=5, shape=())
    initial_angle = float(
        np.pi / 2 * initial_quadrant + np.pi / 4 + (random.uniform(k2) - 0.5) * np.pi / 4
    )
    initial_radius = float(1.0 + random.uniform(k3) * (np.sqrt(2) - 1.0))
    initial_state = jnp.array(
        [initial_radius * np.cos(initial_angle), initial_radius * np.sin(initial_angle)]
    )

    x, u, z, p, dkeys, dvalues, planner_data, planner_data_keys = sim.execute(
        x0=initial_state,
        dynamics=dynamics,
        sensor=sensor,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=setup.dt,
        sigma=setup.R,
        num_steps=N_STEPS,
    )

    success = jnp.linalg.norm(x[-1]) < setup.goal_radius
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

    if completion_times:
        print(f"Avg. Completion Time: {np.mean(completion_times):.2f}")
    else:
        print("Avg. Completion Time: N/A")

    save_data = {
        "x": states,
        "u": controls,
        "z": estimates,
        "p": covariances,
        "pg": setup.pg,
        "successes": successes,
    }

    # Save data to file
    os.makedirs(os.path.dirname(setup.pkl_file), exist_ok=True)
    with open(setup.pkl_file, "wb") as file:
        pickle.dump(save_data, file)

    # Visualizations — save animations to files instead of opening browser
    if setup.VISUALIZE:
        results_dir = "examples/single_integrator/reach_goal/results"
        os.makedirs(results_dir, exist_ok=True)
        print(f"\nSaving animations to {results_dir}/")
        for trial_no in range(setup.n_trials):
            animate(
                states=states[trial_no],
                estimates=estimates[trial_no],
                desired_state=setup.desired_state,
                desired_state_radius=setup.goal_radius,
                x_lim=(-2, 2),
                y_lim=(-2, 2),
                dt=setup.dt,
                title=f"UKF Trial {trial_no + 1}",
                save_animation=True,
                animation_filename=f"{results_dir}/ukf_trial_{trial_no + 1}",
            )
