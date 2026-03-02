import sys
import os

# Add the project root to the path so we can import examples
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array, jacfwd, random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import cbfkit.simulation.simulator as sim

# import cbfkit.controllers.cbf_clf.risk_aware_cbf_clf_controllers as ra_controllers
from cbfkit.controllers.cbf_clf.risk_aware_path_integral_cbf_clf_qp_control_laws import (
    risk_aware_path_integral_cbf_clf_qp_controller as pi_cbf_clf_controller,
)
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.estimators import ct_ekf_dtmeas
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.simulation.monte_carlo import conduct_monte_carlo
from cbfkit.systems import single_integrator
from cbfkit.systems.single_integrator.certificates.lyapunov_function_catalog import position
from cbfkit.utils.user_types import PlannerData, CertificateCollection
from examples.single_integrator.common.config import ekf_state_estimation as setup
from examples.single_integrator.common.lyapunov_functions import fxts_lyapunov_conditions
# from examples.van_der_pol.visualizations.path import animate

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
        conditions=cond_data[0]
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
    def init():
        trajectory.set_data([], [])
        etrajectory.set_data([], [])
        return (trajectory,)

    def update(frame):
        trajectory.set_data(states[:frame, 0], states[:frame, 1])
        etrajectory.set_data(estimates[:frame, 0], estimates[:frame, 1])
        _ = states[frame]
        _ = estimates[frame]
        return (
            trajectory,
            etrajectory,
        )

    fig, ax = plt.subplots()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.plot(desired_state[0], desired_state[1], "ro", markersize=5, label="desired_state")

    ax.add_patch(
        plt.Circle(
            desired_state,
            desired_state_radius,
            color="r",
            fill=False,
            linestyle="--",
            linewidth=1,
        )
    )

    (trajectory,) = ax.plot([], [], label="Trajectory")
    (etrajectory,) = ax.plot([], [], label="Estimated Trajectory")

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid()

    ani = FuncAnimation(fig, update, frames=len(states), init_func=init, blit=True, interval=10)

    if save_animation:
        ani.save(animation_filename, writer="imagemagick", fps=15)

    plt.show()

    return fig, ax


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


def dfdx(x):
    return jacfwd(dynamics)(x)


def h(x):
    return x


def dhdx(_x):
    return jnp.eye((len(setup.desired_state)))


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
ra_clf_params = RiskAwareParams(
    t_max=setup.Tg,
    p_bound=setup.pg,
    gamma=setup.gamma_v,
    eta=setup.eta_v,
    sigma=lambda x: jnp.sqrt(setup.Q),
    varsigma=lambda x: jnp.sqrt(setup.R),
)

controller = pi_cbf_clf_controller(
    dynamics_func=dynamics,
    lyapunovs=lyapunovs,
    control_limits=setup.actuation_limits,
    alpha=jnp.array([0.1]),
    ra_clf_params=ra_clf_params,
    dt=setup.dt,
)


def execute(
    ii: int = 0,
) -> Tuple[Array, Array, Array, Array, List[str], List[Array], List[str], List[Array], bool, float]:
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
        nominal_controller=nominal_controller,
        controller=controller,
        estimator=estimator,
        integrator=integrator,
        dt=setup.dt,
        sigma=setup.R,
        # R=setup.R, # sim.execute doesn't take R directly? Check signature.
        num_steps=N_STEPS,
        # key=key, # sim.execute doesn't take key directly?
        planner_data=PlannerData(x_traj=jnp.zeros((2, N_STEPS + 1))),
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
    os.makedirs(os.path.dirname(setup.pkl_file), exist_ok=True)
    with open(setup.pkl_file, "wb") as file:
        pickle.dump(save_data, file)

    # Visualizations
    if setup.VISUALIZE:
        # Use local animate
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
                animation_filename="examples/single_integrator/reach_goal/results/ekf_estimation",
            )
