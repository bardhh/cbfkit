"""Monte Carlo evaluation of risk-aware CBF-CLF control for unicycle reach-goal."""
import os
import pickle
from typing import List, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array, random

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle as unicycle
from cbfkit.certificates import concatenate_certificates
from cbfkit.certificates.conditions.lyapunov_conditions.fixed_time_stability import fxt_s
from cbfkit.controllers.cbf_clf import (
    risk_aware_cbf_clf_qp_controller as cbf_controller,  # adaptive_cbf_clf_controller,; adaptive_risk_aware_cbf_clf_controller as cbf_controller,; adaptive_stochastic_cbf_controller as cbf_controller,; cbf_clf_controller,; stochastic_cbf_controller,
)
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems.unicycle.models.olfatisaber2002approximate.certificates.lyapunov_functions.reach import (
    reach_goal,
)
from cbfkit.utils import print_progress
from cbfkit.utils.user_types import PlannerData

approx_unicycle_dynamics = unicycle.approx_unicycle_dynamics(lam=1.0)
desired_state = jnp.array([0.0, 0.0, 0])

Q = 0.5 * jnp.eye(3)
ra_clf_params = RiskAwareParams(
    t_max=2.0,  # tf
    p_bound=0.1,
    gamma=0.5,
    eta=10.0,
    sigma=lambda x: Q,
)

# approx_uniycle_nom_controller = unicycle.approx_unicycle_nominal_controller(
#     dynamics=approx_unicycle_dynamics, Kp_pos=1, Kp_theta=0.01, desired_state=desired_state
# )
approx_uniycle_nom_controller = unicycle.zero_controller()
ACTUATION_LIMITS = jnp.array([1e3, 1e3])
goal_radius = 0.1
lyapunov_certificates = concatenate_certificates(
    reach_goal(
        certificate_conditions=fxt_s(c1=2.0, c2=2.0, e1=0.9, e2=1.1),
        goal=desired_state,
        radius=goal_radius,
    )
)
controller = cbf_controller(
    control_limits=ACTUATION_LIMITS,
    nominal_input=approx_uniycle_nom_controller,
    dynamics_func=approx_unicycle_dynamics,
    # barriers=unicycle.barrier_funcs,
    lyapunovs=lyapunov_certificates,
    ra_clf_params=ra_clf_params,
    alpha=jnp.array([1.0] * 3),
)
lyapunov_function = lyapunov_certificates.functions[0]

tf = 2.0 if not os.getenv("CBFKIT_TEST_MODE") else 0.1
dt = 0.01
X_MAX = 5.0
Y_MAX = 5.0
N_TRIALS = 10 if not os.getenv("CBFKIT_TEST_MODE") else 2
N_STEPS = int(tf / dt)
N_STATES = len(desired_state)
N_CONTROLS = approx_uniycle_nom_controller(0.0, desired_state, random.PRNGKey(0), None)[0].shape[0]
filepath = f"examples/unicycle/reach_goal/results/monte_carlo_N{N_TRIALS}.pkl"
progress_update_percent = 10


def compute_lyapunov_trajectory(states: Array) -> Array:
    times = dt * jnp.arange(states.shape[0])
    return jnp.array([lyapunov_function(t, x) for t, x in zip(times, states)])


def compute_covariance_trace_trajectory(covariances: Array) -> Array:
    return jnp.trace(covariances, axis1=1, axis2=2)


# Define simulation function, including post-processing of data
def execute_simulation(
    ii: int,
) -> Tuple[Array, Array, Array, Array, List[str], List[Array], List[str], List[Array]]:
    # Generate random initial condition (position within 5x5 box, angle within +/- pi/4 of heading to goal)
    key = random.PRNGKey(ii)
    k1, k2, k3 = random.split(key, 3)
    x_rand = random.uniform(k1, shape=(), minval=-X_MAX, maxval=X_MAX)
    y_rand = random.uniform(k2, shape=(), minval=-Y_MAX, maxval=Y_MAX)
    a_rand = jnp.arctan2(desired_state[1] - y_rand, desired_state[0] - x_rand) + random.uniform(
        k3, shape=(), minval=-jnp.pi / 4, maxval=jnp.pi / 4
    )
    init_state = jnp.array([x_rand, y_rand, a_rand])

    # Execute simulation
    (
        states,
        controls,
        estimates,
        covariances,
        data,
        data_keys,
        planner_data,
        planner_data_keys,
    ) = sim.execute(
        x0=init_state,  # Changed from initial_conditions.initial_state
        dynamics=approx_unicycle_dynamics,
        sensor=sensor,
        controller=controller,
        nominal_controller=approx_uniycle_nom_controller,
        estimator=estimator,
        integrator=integrator,
        dt=dt,
        num_steps=N_STEPS,  # Changed from n_steps
        planner_data=PlannerData(
            u_traj=None,
            x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, N_STEPS + 1)),
            prev_robustness=None,
        ),
    )

    # Format data as jax numpy arrays
    # w_vals = jnp.array([d[bicycle_data_keys.index("w")] for d in bicycle_data_values])

    print_progress(ii, N_TRIALS)

    return (
        states,
        controls,
        estimates,
        covariances,
        data,
        data_keys,
        planner_data,
        planner_data_keys,
    )


# Needed for multiprocessing
if __name__ == "__main__":
    simulate = 1
    plot = 1 if not os.getenv("CBFKIT_TEST_MODE") else 0

    if simulate:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Process the items sequentially to avoid JAX/multiprocessing issues
        results = [execute_simulation(ii) for ii in range(N_TRIALS)]

        # Convert the results to a NumPy array
        state_record = [result[0] for result in results]
        control_record = [result[1] for result in results]
        estimate_record = [result[2] for result in results]
        covariance_record = [result[3] for result in results]
        lyapunov_record = [compute_lyapunov_trajectory(states) for states in state_record]
        covariance_trace_record = [
            compute_covariance_trace_trajectory(covariances) for covariances in covariance_record
        ]

        # Convert data to dict object
        save_data = {}
        save_keys = [
            "state_record",
            "control_record",
            "estimate_record",
            "covariance_record",
            "lyapunov_record",
            "covariance_trace_record",
        ]
        for i, array in enumerate(
            [
                state_record,
                control_record,
                estimate_record,
                covariance_record,
                lyapunov_record,
                covariance_trace_record,
            ]
        ):
            save_data[save_keys[i]] = array

        # Save data in pickle format
        with open(filepath, "wb") as file:
            pickle.dump(save_data, file)

        state_trajectories = state_record
        control_trajectories = control_record
        lyapunov_trajectories = lyapunov_record
        covariance_trace_trajectories = covariance_trace_record

    else:
        # Load data from file
        with open(filepath, "rb") as read_file:
            loaded_data = pickle.load(read_file)

        print("Data Loaded.")

        state_trajectories = loaded_data["state_record"]
        control_trajectories = loaded_data["control_record"]
        lyapunov_trajectories = loaded_data.get("lyapunov_record")
        covariance_trace_trajectories = loaded_data.get("covariance_trace_record")

        # Backward compatibility with older pickle format where these fields were mislabeled.
        if lyapunov_trajectories is None or (
            len(lyapunov_trajectories) > 0 and lyapunov_trajectories[0].ndim != 1
        ):
            lyapunov_trajectories = [
                compute_lyapunov_trajectory(states) for states in state_trajectories
            ]
        if covariance_trace_trajectories is None:
            covariance_trajectories = loaded_data.get(
                "covariance_record", loaded_data.get("w_record")
            )
            if covariance_trajectories is not None:
                covariance_trace_trajectories = [
                    compute_covariance_trace_trajectory(covariances)
                    for covariances in covariance_trajectories
                ]
            else:
                covariance_trace_trajectories = []

    if plot:
        n_trials = len(state_trajectories)
        fig, ax = plt.subplots()
        for states in state_trajectories:
            ax.plot(states[:, 0], states[:, 1], alpha=0.5)
        ax.plot(desired_state[0], desired_state[1], "ro", markersize=5, label="desired_state")
        ax.add_patch(
            plt.Circle(
                (desired_state[0], desired_state[1]),
                goal_radius,
                color="r",
                fill=False,
                linestyle="--",
                linewidth=1,
            )
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("State Trajectories")
        ax.grid(True)
        ax.axis("equal")
        ax.legend()

        fig2, ax2 = plt.subplots()
        for ii in range(n_trials):
            if lyapunov_trajectories[ii] is not None and len(lyapunov_trajectories[ii]) > 0:
                ax2.plot(lyapunov_trajectories[ii], alpha=0.5)
        ax2.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax2.set_title("Lyapunov Value")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("V(x)")
        ax2.grid(True)

        fig3, (ax3, ax4) = plt.subplots(2, 1, sharex=True)
        for ii in range(n_trials):
            if control_trajectories[ii] is not None:
                ax3.plot(control_trajectories[ii][:, 0], alpha=0.5)
                ax4.plot(control_trajectories[ii][:, 1], alpha=0.5)
        ax3.set_ylabel("u[0]")
        ax3.set_title("Control Inputs")
        ax3.grid(True)
        ax4.set_ylabel("u[1]")
        ax4.set_xlabel("Step")
        ax4.grid(True)

        fig4, ax5 = plt.subplots()
        for ii in range(n_trials):
            if (
                covariance_trace_trajectories[ii] is not None
                and len(covariance_trace_trajectories[ii]) > 0
            ):
                ax5.plot(covariance_trace_trajectories[ii], alpha=0.5)
        ax5.set_title("Estimator Covariance Trace")
        ax5.set_xlabel("Step")
        ax5.set_ylabel("trace(P)")
        ax5.grid(True)

        plt.show()

    n_trials = len(state_trajectories)
    final_deviation = jnp.array([traj[-1, :2] - desired_state[:2] for traj in state_trajectories])
    success_fraction = jnp.sum(jnp.linalg.norm(final_deviation, axis=1) < 0.25) / n_trials
    print(f"Success Fraction: {success_fraction:.2f}")


# plot = 0
# animate = 1

# if plot:
#     unicycle.plot_trajectory(
#         states=bicycle_states,
#         desired_state=desired_state,
#         desired_state_radius=0.1,
#         x_lim=(-4, 4),
#         y_lim=(-4, 4),
#         title="System Behavior",
#     )

# if animate:
#     unicycle.animate(
#         states=bicycle_states,
#         desired_state=desired_state,
#         desired_state_radius=0.1,
#         x_lim=(-4, 4),
#         y_lim=(-4, 4),
#         dt=dt,
#         title="System Behavior",
#     )
