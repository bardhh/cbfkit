import os
import pickle
from typing import Any, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array, jacfwd, random

import cbfkit.simulation.simulator as sim

# Load dynamics, sensors, controller, estimator, integrator
import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
from cbfkit.certificates import concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.certificates.conditions.lyapunov_conditions.exponential_stability import e_s

# Import controller functions
from cbfkit.controllers.cbf_clf.risk_aware_path_integral_cbf_clf_qp_control_laws import (
    risk_aware_path_integral_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.estimators import ct_ekf_dtmeas as ekf
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import unbiased_gaussian_noise as sensor

# Load initial conditions
from cbfkit.utils.user_types import (
    Control,
    ControllerCallable,
    ControllerData,
    DynamicsCallable,
    EstimatorCallable,
    IntegratorCallable,
    Key,
    NominalControllerCallable,
    PerturbationCallable,
    PlannerCallable,
    PlannerData,
    SensorCallable,
    State,
)
from examples.unicycle.common.config import ekf_state_estimation as initial_conditions
from examples.unicycle.common.visualizations import plot_trajectory

# Whether or not to simulate, plot
plot = 1
save = 1
save_path = "examples/unicycle/reach_goal/results/"
file_name = os.path.basename(__file__)[:-8]

# Define time parameters
tf = 2.0
dt = 0.01
n_steps = int(tf / dt)

X_MAX = 5.0
Y_MAX = 5.0
if os.environ.get("CBFKIT_TEST_MODE") == "true":
    N_TRIALS = 2
else:
    N_TRIALS = 50
N_STEPS = int(tf / dt)
N_STATES = len(initial_conditions.desired_state)
N_CONTROLS = 2

# Define dynamics, controller, and estimator with specified parameters
approx_unicycle_dynamics = unicycle.plant(lam=1.0)


def dfdx(x):
    return jacfwd(approx_unicycle_dynamics)(x)


def h(x):
    return x


def dhdx(x):
    return jnp.eye((len(initial_conditions.initial_state)))


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
    unicycle.certificates.barrier_functions.obstacle_ca(
        certificate_conditions=zeroing_barriers.linear_class_k(1.0),
        obstacle=obs,
        ellipsoid=ell,
    )
    for obs, ell in zip(obstacles, ellipsoids)
]
barrier_packages = concatenate_certificates(*barriers)
risk_aware_barrier_params = RiskAwareParams(
    t_max=tf,
    p_bound=0.05,
    gamma=0.75,
    eta=10.0,
    sigma=lambda _: 1 * initial_conditions.R,
)

# Lyapunov function configuration
lyapunovs = [
    unicycle.certificates.lyapunov_functions.reach_goal(
        certificate_conditions=e_s(1.0),
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


# N_TRIALS: num_sims
# N_STEPS: num_steps
# N_STATES: n_states
# N_CONTROLS: n_controls

# Define variables for execute function to access
n_sims = N_TRIALS
num_steps = N_STEPS
N = N_STEPS

# x0s for monte carlo simulation
x0s = [initial_conditions.initial_state for _ in range(N_TRIALS)]

# Optional arguments for sim.execute (if not defined elsewhere)
planner = None
perturbation = None
sigma = initial_conditions.R


# Define simulation function, including post-processing of data
def execute_trial(
    ii: int,
    n_sims: int,
    x0s: List[Array],
    dt: float,
    num_steps: int,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    planner: Optional[PlannerCallable],
    nominal_controller: Optional[NominalControllerCallable],
    controller: Optional[ControllerCallable],
    sensor: SensorCallable,
    estimator: EstimatorCallable,
    perturbation: Optional[PerturbationCallable],
    sigma: Optional[Array],
    N: int,
    initial_conditions: Any,
    tf: float,
) -> Tuple[Array, Array, Array, Array, List[str], List[Array], List[str], List[Array]]:
    print(f"Simulation {ii+1}/{n_sims}")
    x0 = x0s[ii]
    # Pass a unique key for each trial based on the index ii
    # This ensures deterministic reproducibility for the entire script run
    # while providing different noise sequences for each trial.
    key = random.PRNGKey(ii)
    x, u, z, p, data_keys, data_values, planner_keys, planner_values = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=num_steps,
        dynamics=dynamics,
        integrator=integrator,
        planner=planner,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        perturbation=perturbation,
        sigma=sigma,
        key=key,
        planner_data=sim.PlannerData(
            u_traj=jnp.zeros((2, N)),
            x_traj=jnp.tile(initial_conditions.desired_state.reshape(-1, 1), (1, int(tf / dt) + 1)),
            prev_robustness=None,
        ),
        use_jit=True,
    )

    return x, u, z, p, data_keys, data_values, planner_keys, planner_values


# Instantiate the controller
controller_instance = cbf_controller(
    control_limits=initial_conditions.actuation_limits,
    dynamics_func=approx_unicycle_dynamics,
    barriers=barrier_packages,
    lyapunovs=lyapunov_packages,
    ra_cbf_params=risk_aware_barrier_params,
    ra_clf_params=risk_aware_lyapunov_params,
    nominal_input=nominal_controller,
)

# Needed for multiprocessing
if __name__ == "__main__":
    import pickle

    simulate = 1
    plot = 1

    if simulate:
        # Execute trials sequentially
        results = []
        for ii in range(N_TRIALS):
            results.append(
                execute_trial(
                    ii=ii,
                    n_sims=n_sims,
                    x0s=x0s,
                    dt=dt,
                    num_steps=num_steps,
                    dynamics=approx_unicycle_dynamics,
                    integrator=integrator,
                    planner=planner,
                    nominal_controller=nominal_controller,
                    controller=controller_instance,
                    sensor=sensor,
                    estimator=estimator,
                    perturbation=perturbation,
                    sigma=sigma,
                    N=N,
                    initial_conditions=initial_conditions,
                    tf=tf,
                )
            )

        # Convert the results to a NumPy array
        state_record = [result[0] for result in results]
        control_record = [result[1] for result in results]

        # Convert data to dict object
        save_data = {}
        save_keys = ["state_record", "control_record", "lyapunov_record", "w_record"]
        for i, array in enumerate([state_record, control_record]):
            save_data[save_keys[i]] = array

        # Save data in pickle format
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path + file_name + ".pkl", "wb") as f_save:
            pickle.dump(save_data, f_save)

    else:
        # Load data from file
        with open(save_path + file_name + ".pkl", "rb") as f_load:
            loaded_data = pickle.load(f_load)

        print("Data Loaded.")

        state_record = loaded_data["state_record"]
        control_record = loaded_data["control_record"]

    if plot:
        from examples.unicycle.common.visualizations import plot_trajectory

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

        # fig, ax = plt.subplots()
        # for states in state_record:
        #     ax.plot(states[:, 0], states[:, 1])
        # for controls in control_record:
        #     ax.plot(jnp.linspace(0.0, tf, N_STEPS), controls[:, 0])
        #     ax.plot(jnp.linspace(0.0, tf, N_STEPS), controls[:, 1])
        fig.savefig(save_path + file_name + ".png")
        print(f"Figure saved to: {save_path + file_name}.png")

    final_deviation = jnp.array(
        [rec[-1, :2] - initial_conditions.desired_state[:2] for rec in state_record]
    )
    success_fraction = jnp.sum(jnp.linalg.norm(final_deviation, axis=1) < 0.25) / N_TRIALS
    print(f"Success Fraction: {success_fraction:.2f}")
