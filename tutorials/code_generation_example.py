"""Tutorial: Code generation for dynamical systems.

Script equivalent of `code_generation_tutorial.ipynb`. Demonstrates how to use
cbfkit.codegen to generate a new dynamical system model and simulate it with
CBF-CLF safety constraints.
"""

import os
import importlib
import jax.numpy as jnp
from cbfkit.codegen.create_new_system import generate_model
import cbfkit.simulation.simulator as sim
import cbfkit.controllers.cbf_clf as cbf_clf_controllers
from cbfkit.certificates import concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.certificates.conditions.lyapunov_conditions.exponential_stability import e_s
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation


def sigma(x):
    """
    Computes the state-dependent diffusion term for the stochastic differential equation (SDE).

    Args:
        x (Array): The current state vector.

    Returns:
        Array: The diffusion matrix.
    """
    return jnp.array([[0, 0], [0, 0.05 * x[0]]])


def main():
    # =========================================================================
    # 1. Model Generation
    # =========================================================================
    print("Generating model...")

    # Define dynamics
    drift_dynamics = "[x[1], -x[0] + epsilon * (1 - x[0]**2) * x[1]]"
    control_matrix = "[[0], [1]]"

    # Define target directory and model name
    # We assume the script is run from the repository root, so generated files go to tutorials/
    target_directory = "./tutorials"
    model_name = "van_der_pol_oscillator"

    params = {"dynamics": {"epsilon: float": 0.5}}

    # Define nominal controller
    nominal_control_law = "x[0] * (1 - k_p) - epsilon * (1 - x[0]**2) * x[1]"
    params["controller"] = {"k_p: float": 1.0, "epsilon: float": 0.5}

    # Define constraints
    state_constraint_funcs = ["5 - x[0]", "x[0] + 7"]
    lyapunov_functions = "x[0]**2 + x[1]**2 - radius"
    params["clf"] = [{"radius: float": 1.0}]

    # Generate model
    generate_model.generate_model(
        directory=target_directory,
        model_name=model_name,
        drift_dynamics=drift_dynamics,
        control_matrix=control_matrix,
        barrier_funcs=state_constraint_funcs,
        lyapunov_funcs=lyapunov_functions,
        nominal_controller=nominal_control_law,
        params=params,
    )

    print(f"Model generated in {target_directory}/{model_name}")

    # =========================================================================
    # 2. Simulation Setup
    # =========================================================================
    print("Setting up simulation...")

    # Dynamically import the generated module
    # We use importlib to ensure we can import the newly generated package
    try:
        from tutorials import van_der_pol_oscillator

        importlib.reload(van_der_pol_oscillator)
    except ImportError:
        # Fallback if tutorials is not in path or other issues
        import sys

        sys.path.append(os.path.abspath(target_directory))
        import van_der_pol_oscillator

    SAVE_FILE = f"{target_directory}/{model_name}/simulation_data"
    DT = 1e-2
    TF = 10.0 if not os.getenv("CBFKIT_TEST_MODE") else 1.0
    N_STEPS = int(TF / DT) + 1
    INITIAL_STATE = jnp.array([1.5, 0.25])
    ACTUATION_LIMITS = jnp.array([100.0])

    eps = 0.5
    dynamics = van_der_pol_oscillator.plant(
        epsilon=eps, perturbation=generate_stochastic_perturbation(sigma, DT)
    )

    b1 = van_der_pol_oscillator.certificate_functions.barrier_functions.cbf1_package(
        certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0),
    )
    b2 = van_der_pol_oscillator.certificate_functions.barrier_functions.cbf2_package(
        certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0),
    )
    barriers = concatenate_certificates(b1, b2)

    l1 = van_der_pol_oscillator.certificate_functions.lyapunov_functions.clf1_package(
        certificate_conditions=e_s(c=2.0),
        radius=1.0,
    )
    lyapunov = concatenate_certificates(l1)

    nominal_controller = van_der_pol_oscillator.controllers.controller_1(k_p=1.0, epsilon=eps)

    cbf_clf_controller = cbf_clf_controllers.vanilla_cbf_clf_qp_controller(
        control_limits=ACTUATION_LIMITS,
        dynamics_func=dynamics,
        barriers=barriers,
        lyapunovs=lyapunov,
        relaxable_clf=True,
    )

    # =========================================================================
    # 3. Simulation Execution
    # =========================================================================
    print("Starting simulation...")

    sim.execute(
        x0=INITIAL_STATE,
        dt=DT,
        num_steps=N_STEPS,
        dynamics=dynamics,
        perturbation=generate_stochastic_perturbation(sigma=sigma, dt=DT),
        integrator=integrator,
        planner=None,
        nominal_controller=nominal_controller,
        controller=cbf_clf_controller,
        sensor=sensor,
        estimator=estimator,
        filepath=SAVE_FILE,
        planner_data={
            "x_traj": jnp.zeros((2, 1)),
        },
        controller_data={},
    )

    print(f"Simulation complete. Data saved to {SAVE_FILE}.csv")


if __name__ == "__main__":
    main()
