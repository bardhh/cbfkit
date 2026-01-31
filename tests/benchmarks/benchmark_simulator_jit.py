import os
import sys
import time
import jax.numpy as jnp

# Add the project root directory to the python path
sys.path.append(os.getcwd() + "/src")
sys.path.append(os.getcwd())

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.utils.user_types import PlannerData
from examples.unicycle.common.ellipsoidal_obstacle import cbf as ellipsoid_cbf

def run_benchmark():
    tf = 10.0
    dt = 0.01

    init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
    desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
    actuation_constraints = jnp.array([100.0, 100.0])

    unicycle_dynamics = unicycle.plant(lam=1.0)
    unicycle_dynamics.a_max = actuation_constraints[0]
    unicycle_dynamics.omega_max = actuation_constraints[1]
    unicycle_dynamics.v_max = 1.0
    unicycle_dynamics.goal_tol = 0.25

    uniycle_nom_controller = unicycle.controllers.proportional_controller(
        dynamics=unicycle_dynamics,
        Kp_pos=1.0,
        Kp_theta=1.0,
    )

    # Add more obstacles to make it heavier
    obstacles = [
        (1, 2.0, 0.0),
        (3.0, 2.0, 0.0),
        (-1.0, 1.0, 0.0),
        (0.5, -1.0, 0.0),
    ] * 5 # Increase load

    ellipsoids = [
        (0.5, 1.5),
        (0.75, 2.0),
        (1.0, 0.75),
        (0.75, 0.5),
    ] * 5

    barriers = [
        rectify_relative_degree(
            function=ellipsoid_cbf(jnp.array(obs), jnp.array(ell)),
            system_dynamics=unicycle_dynamics,
            state_dim=len(init_state),
            form="exponential",
        )(
            certificate_conditions=zeroing_barriers.linear_class_k(5.0),
            obstacle=jnp.array(obs),
            ellipsoid=jnp.array(ell),
        )
        for obs, ell in zip(obstacles, ellipsoids)
    ]

    barrier_packages = concatenate_certificates(*barriers)

    controller = cbf_controller(
        control_limits=actuation_constraints,
        dynamics_func=unicycle_dynamics,
        barriers=barrier_packages,
    )

    print("Warming up...")
    sim.execute(
        x0=init_state,
        dt=dt,
        num_steps=10,
        dynamics=unicycle_dynamics,
        integrator=integrator,
        nominal_controller=uniycle_nom_controller,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        verbose=False,
        planner_data=PlannerData(
            x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, 11)),
            prev_robustness=None,
        ),
        use_jit=True,
    )

    print("Running benchmark...")
    start_time = time.time()
    n_runs = 5
    for i in range(n_runs):
        sim.execute(
            x0=init_state,
            dt=dt,
            num_steps=int(tf / dt),
            dynamics=unicycle_dynamics,
            integrator=integrator,
            nominal_controller=uniycle_nom_controller,
            controller=controller,
            sensor=sensor,
            estimator=estimator,
            verbose=False,
            planner_data=PlannerData(
                x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, int(tf / dt) + 1)),
                prev_robustness=None,
            ),
            use_jit=True,
        )
    end_time = time.time()
    avg_time = (end_time - start_time) / n_runs
    print(f"Average time per run: {avg_time:.4f}s")

if __name__ == "__main__":
    run_benchmark()
