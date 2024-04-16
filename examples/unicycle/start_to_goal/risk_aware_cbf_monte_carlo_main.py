"""
This module provides a template for example simulations designed to be run from
the interpreter via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the
src/cbfkit tree.

"""

import os
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, Array
from typing import List

# Whether or not to simulate, plot
plot = 1
save = 1
save_path = "examples/unicycle/start_to_goal/results/"  # nominally_controlled/ekf_state_estimation/results/"
file_name = os.path.basename(__file__)[:-8]

# Load system module
import cbfkit.simulation.simulator as sim

# Load dynamics, sensors, controller, estimator, integrator
import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle

from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.estimators import ct_ekf_dtmeas as ekf
from cbfkit.integration import forward_euler as integrator

# Import controller functions
from cbfkit.controllers.model_based.cbf_clf_controllers.risk_aware_path_integral_cbf_clf_qp_control_laws import (
    risk_aware_path_integral_cbf_clf_qp_controller,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
    e_s,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.risk_aware_params import (
    RiskAwareParams,
)

# Load initial conditions
from examples.unicycle.start_to_goal.initial_conditions import (
    ekf_state_estimation as initial_conditions,
)

# Define time parameters
tf = 2.0
dt = 0.01
n_steps = int(tf / dt)

X_MAX = 5.0
Y_MAX = 5.0
N_TRIALS = 5
N_STEPS = int(tf / dt)
N_STATES = len(initial_conditions.desired_state)
N_CONTROLS = 2

# Define dynamics, controller, and estimator with specified parameters
approx_unicycle_dynamics = unicycle.plant(l=1.0)
dfdx = lambda x: jacfwd(approx_unicycle_dynamics)(x)
h = lambda x: x
dhdx = lambda x: jnp.eye((len(initial_conditions.initial_state)))
nominal_controller = unicycle.controllers.zero_controller()
scale_factor = 1.25
estimator = ekf(
    Q=initial_conditions.Q * scale_factor,
    R=initial_conditions.R * scale_factor,
    dynamics=approx_unicycle_dynamics,
    dfdx=dfdx,
    h=h,
    dhdx=dhdx,
    dt=dt,
)

# Barrier function configuration
obstacles = [
    (1.0, 2.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.5, -1.0, 0.0),
]
ellipsoids = [
    (0.5, 1.5),
    (1.0, 0.75),
    (0.75, 0.5),
]
barriers = [
    unicycle.certificate_functions.barrier_functions.obstacle_ca(
        certificate_conditions=zeroing_barriers.linear_class_k(2.0),
        obstacle=obs,
        ellipsoid=ell,
    )
    for obs, ell in zip(obstacles, ellipsoids)
]
barrier_packages = concatenate_certificates(*barriers)
risk_aware_barrier_params = RiskAwareParams(
    t_max=tf,
    p_bound=0.1,
    gamma=0.1,
    eta=2.0,
    sigma=lambda _: initial_conditions.R,
)

# Lyapunov function configuration
lyapunovs = [
    unicycle.certificate_functions.lyapunov_functions.reach_goal(
        certificate_conditions=e_s(2.0),
        goal=initial_conditions.desired_state,
        radius=0.1,
    )
]
lyapunov_packages = concatenate_certificates(*lyapunovs)
risk_aware_lyapunov_params = RiskAwareParams(
    t_max=tf,
    p_bound=0.99,
    gamma=10.0,
    eta=2.0,
    sigma=lambda _: initial_conditions.R,
)

controller = risk_aware_path_integral_cbf_clf_qp_controller(
    control_limits=jnp.array([100.0, 100.0]),
    nominal_input=nominal_controller,
    dynamics_func=approx_unicycle_dynamics,
    barriers=barrier_packages,
    lyapunovs=lyapunov_packages,
    ra_cbf_params=risk_aware_barrier_params,
    ra_clf_params=risk_aware_lyapunov_params,
)


# Define simulation function, including post-processing of data
def execute_simulation(ii: int) -> List[Array]:
    # Generate random initial condition (position within 5x5 box, angle within +/- pi/4 of heading to goal)
    np.random.seed(ii)
    x_rand = np.random.uniform(low=-X_MAX, high=X_MAX)
    y_rand = np.random.uniform(low=-Y_MAX, high=Y_MAX)
    a_rand = jnp.arctan2(
        initial_conditions.desired_state[1] - y_rand, initial_conditions.desired_state[0] - x_rand
    ) + np.random.uniform(low=-jnp.pi / 4, high=jnp.pi / 4)
    init_state = jnp.array([x_rand, y_rand, a_rand])

    # Execute simulation
    x, u, z, p, data, data_keys = sim.execute(
        x0=init_state,
        dt=dt,
        num_steps=int(tf / dt),
        dynamics=approx_unicycle_dynamics,
        integrator=integrator,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        sigma=initial_conditions.R,
        filepath=save_path + file_name,
    )

    return [x, u]


# Needed for multiprocessing
if __name__ == "__main__":
    import pickle
    import multiprocessing as mp
    import matplotlib.pyplot as plt

    simulate = 1
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

        print(state_record.shape)

        # Convert data to dict object
        save_data = {}
        save_keys = ["state_record", "control_record", "lyapunov_record", "w_record"]
        for i, array in enumerate([state_record, control_record]):
            save_data[save_keys[i]] = array

        # Save data in pickle format
        with open(save_path + file_name + ".pkl", "wb") as file:
            pickle.dump(save_data, file)

    else:
        # Load data from file
        with open(save_path + file_name + ".pkl", "rb") as file:
            loaded_data = pickle.load(file)

        print("Data Loaded.")

        state_record = loaded_data["state_record"]
        control_record = loaded_data["control_record"]

    if plot:
        from examples.unicycle.start_to_goal.visualizations import plot_trajectory

        fig, ax = plt.subplots()

        for states in state_record:
            fig, ax = plot_trajectory(
                states=states,
                desired_state=initial_conditions.desired_state,
                desired_state_radius=0.1,
                obstacles=obstacles,
                ellipsoids=ellipsoids,
                x_lim=(-2, 6),
                y_lim=(-2, 6),
                title="System Behavior",
            )

        fig, ax = plt.subplots()
        for states in state_record:
            print(states[:, 0])
            print(states[:, 1])
            ax.plot(states[:, 0], states[:, 1])
        fig.savefig(save_path + file_name + ".png")

    final_deviation = np.array(state_record[:, -1, :2] - initial_conditions.desired_state[:2])
    success_fraction = (
        len(final_deviation[np.where(jnp.linalg.norm(final_deviation, axis=1) < 0.25)]) / N_TRIALS
    )
    print(f"Success Fraction: {success_fraction:.2f}")
