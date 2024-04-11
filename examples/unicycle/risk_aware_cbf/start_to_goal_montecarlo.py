import jax.numpy as jnp
import pickle
from jax import random, Array
import multiprocessing as mp
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from cbfkit.utils import print_progress

import cbfkit.models.unicycle as unicycle
import cbfkit.system as system
from cbfkit.cbf_clf_controllers import (
    # adaptive_cbf_clf_controller,
    # adaptive_risk_aware_cbf_clf_controller as cbf_controller,
    # adaptive_stochastic_cbf_controller as cbf_controller,
    # cbf_clf_controller,
    risk_aware_cbf_clf_controller as cbf_controller,
    # stochastic_cbf_controller,
)

approx_unicycle_dynamics = unicycle.approx_unicycle_dynamics(l=1.0, stochastic=True)
desired_state = jnp.array([0.0, 0.0, 0])

# approx_uniycle_nom_controller = unicycle.approx_unicycle_nominal_controller(
#     dynamics=approx_unicycle_dynamics, Kp_pos=1, Kp_theta=0.01, desired_state=desired_state
# )
approx_uniycle_nom_controller = unicycle.zero_controller()
controller = cbf_controller(
    nominal_input=approx_uniycle_nom_controller,
    dynamics_func=approx_unicycle_dynamics,
    # barriers=unicycle.barrier_funcs,
    lyapunovs=unicycle.fxt_lyapunov_funcs(desired_state),
    alpha=jnp.array([1.0] * 3),
)

tf = 2.0
dt = 0.01
X_MAX = 5.0
Y_MAX = 5.0
N_TRIALS = 50000
N_STEPS = int(tf / dt)
N_STATES = len(desired_state)
N_CONTROLS = approx_uniycle_nom_controller(0.0, desired_state)[0].shape[0]
filepath = f"examples/unicycle/results/monte_carlo_N{N_TRIALS}.pkl"
progress_update_percent = 10


# Define simulation function, including post-processing of data
def execute_simulation(ii: int) -> List[Array]:
    # Generate random initial condition (position within 5x5 box, angle within +/- pi/4 of heading to goal)
    np.random.seed(ii)
    x_rand = np.random.uniform(low=-X_MAX, high=X_MAX)
    y_rand = np.random.uniform(low=-Y_MAX, high=Y_MAX)
    a_rand = jnp.arctan2(desired_state[1] - y_rand, desired_state[0] - x_rand) + np.random.uniform(
        low=-jnp.pi / 4, high=jnp.pi / 4
    )
    init_state = jnp.array([x_rand, y_rand, a_rand])

    bicycle_states, bicycle_data_keys, bicycle_data_values = system.simulate(
        state=init_state,
        dynamics=approx_unicycle_dynamics,
        # controller=approx_uniycle_nom_controller,
        controller=controller,
        dt=dt,
        num_steps=N_STEPS,
    )

    # Format data as jax numpy arrays
    state = jnp.array(bicycle_states)
    control = jnp.array([d[bicycle_data_keys.index("u")] for d in bicycle_data_values])
    lyapunov = jnp.array([d[bicycle_data_keys.index("clfs")] for d in bicycle_data_values])
    w_vals = jnp.array([d[bicycle_data_keys.index("w")] for d in bicycle_data_values])

    print_progress(ii, N_TRIALS)

    return [state, control, lyapunov, w_vals]


# Needed for multiprocessing
if __name__ == "__main__":
    simulate = 0
    plot = 1

    if simulate:
        # Create a multiprocessing Pool
        with mp.Pool(processes=8) as pool:
            # Process the items in parallel
            results = pool.map(execute_simulation, range(N_TRIALS))

        # Close the pool and wait for the processes to finish
        pool.close()
        pool.join()

        # Convert the results to a NumPy array
        state_record = np.array([result[0] for result in results])
        control_record = np.array([result[1] for result in results])
        lyapunov_record = np.array([result[2] for result in results])
        w_record = np.array([result[3] for result in results])

        # Convert data to dict object
        save_data = {}
        save_keys = ["state_record", "control_record", "lyapunov_record", "w_record"]
        for i, array in enumerate([state_record, control_record, lyapunov_record, w_record]):
            save_data[save_keys[i]] = array

        # Save data in pickle format
        with open(filepath, "wb") as file:
            pickle.dump(save_data, file)

    # Load data from file
    with open(filepath, "rb") as file:
        loaded_data = pickle.load(file)

    print("Data Loaded.")

    state_trajectories = loaded_data["state_record"]
    control_trajectories = loaded_data["control_record"]
    lyapunov_trajectories = loaded_data["lyapunov_record"]
    w_trajectories = loaded_data["w_record"]

    if plot:
        fig, ax = plt.subplots()

        # for states in state_trajectories:
        #     fig, ax = unicycle.plot_trajectory(
        #         fig,
        #         ax,
        #         states=states,
        #         desired_state=desired_state,
        #         desired_state_radius=0.25,
        #         x_lim=(-4, 4),
        #         y_lim=(-2, 6),
        #         title="System Behavior",
        #     )

        # fig2, ax2 = plt.subplots()
        # for ii in range(N_TRIALS):
        #     ax2.plot(w_trajectories[ii, :, 0])

        # fig3, ax3 = plt.subplots()
        # for ii in range(N_TRIALS):
        #     ax3.plot(control_trajectories[ii, :, 0])
        #     ax3.plot(control_trajectories[ii, :, 1])

        plt.show()

    final_deviation = np.array(state_trajectories[:, -1, :2] - desired_state[:2])
    success_fraction = (
        len(final_deviation[np.where(jnp.linalg.norm(final_deviation, axis=1) < 0.25)]) / N_TRIALS
    )
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
