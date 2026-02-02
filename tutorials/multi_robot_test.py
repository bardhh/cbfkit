import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import functools
import importlib
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

import cbfkit.controllers.cbf_clf as cbf_clf_controllers
import cbfkit.simulation.simulator as sim
from cbfkit.codegen.create_new_system import generate_model
from cbfkit.certificates import concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.certificates.conditions.lyapunov_conditions.exponential_stability import e_s
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor

def main():
    target_directory = "./tutorials"
    model_name = "multi_augmented_single_integrators"
    num_robots = 10
    INITIAL_STATE = np.zeros(2 * num_robots)
    goals = np.zeros(2 * num_robots)
    DT = 0.1
    TF = 10
    N_STEPS = int(TF / DT) + 1
    ACTUATION_LIMITS = 100 * jnp.ones(2 * num_robots)

    # Set initial states and goals of all the robots
    # The robot states are appended in a single vector
    for i in range(num_robots):
        theta_disturbance = np.clip(np.random.normal(0, 1.0), -np.pi / 60, np.pi / 60)
        INITIAL_STATE[2 * i] = 2 * np.cos(2 * np.pi * i / num_robots + theta_disturbance)
        INITIAL_STATE[2 * i + 1] = 2 * np.sin(2 * np.pi * i / num_robots + theta_disturbance)
        goals[2 * i] = -2 * np.cos(2 * np.pi * i / num_robots) + 0.1

    params: Dict[str, Any] = {}

    state_constraint_funcs = []
    for i in range(num_robots):
        for j in range(i + 1, num_robots):
            state_constraint_funcs.append(f"(x[{2*i}]-x[{2*j}])**2 + (x[{2*i+1}]-x[{2*j+1}])**2 - 1")

    params["clf"] = []
    lyapunov_functions = []
    for i in range(num_robots):
        lyapunov_functions.append(f"(x[{2*i}]-goal[0])**2+(x[{2*i+1}]-goal[1])**2")
        params["clf"].append(
            {
                "goal: float": goals[2 * i : 2 * i + 2],
            }
        )

    # Dynamics generation parameters
    drift_dynamics_single_robot = np.array([0, 0])
    control_matrix_single_robot = np.eye(2)
    drift_dynamics = np.tile(drift_dynamics_single_robot, num_robots)
    control_matrix = np.kron(np.eye(num_robots), control_matrix_single_robot)
    drift_dynamics_str = np.array2string(drift_dynamics, separator=",").replace("\n", "")
    control_matrix_str = np.array2string(control_matrix, separator=",").replace("\n", "")

    nominal_control_law = "["
    for i in range(num_robots):
        if i < (num_robots - 1):
            nominal_control_law = (
                nominal_control_law
                + f" -k_p * (x[{2*i}]-goal[{2*i}]), -k_p * (x[{2*i+1}]-goal[{2*i+1}]),"
            )
        else:
            nominal_control_law = (
                nominal_control_law
                + f" -k_p * (x[{2*i}]-goal[{2*i}]), -k_p * (x[{2*i+1}]-goal[{2*i+1}])"
            )
    nominal_control_law = nominal_control_law + "]"

    params["controller"] = {
        "goal: float": goals,
        "k_p: float": 1.0,
    }

    # Generate model
    generate_model.generate_model(
        directory=target_directory,
        model_name=model_name,
        drift_dynamics=drift_dynamics_str,
        control_matrix=control_matrix_str,
        barrier_funcs=state_constraint_funcs,
        lyapunov_funcs=lyapunov_functions,
        nominal_controller=nominal_control_law,
        params=params,
    )

    # Invalidate caches to ensure the newly generated module is found
    importlib.invalidate_caches()

    # Dynamically import the generated module
    multi_augmented_single_integrators = importlib.import_module(f"tutorials.{model_name}")

    # Simulation Parameters
    SAVE_FILE = f"tutorials/{model_name}/simulation_data"  # automatically uses .csv format

    bs = []
    for i in range(len(state_constraint_funcs)):
        func_path = f"certificate_functions.barrier_functions.cbf{i + 1}_package"
        func = functools.reduce(getattr, func_path.split("."), multi_augmented_single_integrators)
        bs.append(func(certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0)))
    barriers = concatenate_certificates(*bs)

    ls = []
    for i in range(len(lyapunov_functions)):
        func_path = f"certificate_functions.lyapunov_functions.clf{i + 1}_package"
        func = functools.reduce(getattr, func_path.split("."), multi_augmented_single_integrators)
        ls.append(
            func(
                certificate_conditions=e_s(c=2.0),
                goal=goals[2 * i : 2 * i + 2],
            )
        )
    lyapunov = concatenate_certificates(*ls)

    dynamics = multi_augmented_single_integrators.plant()

    nominal_controller = multi_augmented_single_integrators.controllers.controller_1(goal=goals, k_p=1)

    cbf_clf_controller = cbf_clf_controllers.vanilla_cbf_clf_qp_controller(
        control_limits=ACTUATION_LIMITS,
        nominal_input=nominal_controller,
        dynamics_func=dynamics,
        barriers=barriers,
        lyapunovs=lyapunov,
        relaxable_clf=True,
    )

    x, _u, _z, _p, dkeys, dvalues, pkeys, pvalues = sim.execute(
        x0=jnp.array(INITIAL_STATE),
        dt=DT,
        num_steps=N_STEPS,
        dynamics=dynamics,
        integrator=integrator,
        controller=cbf_clf_controller,
        sensor=sensor,
        estimator=estimator,  # perturbation=generate_stochastic_perturbation(sigma=sigma, dt=DT),
        filepath=SAVE_FILE,
        use_jit=True,
    )

if __name__ == "__main__":
    main()
