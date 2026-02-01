"""
Scout Benchmark: Solver Statistics
==================================

This benchmark runs a standard CBF-CLF-QP simulation (unicycle reaching a goal)
and reports solver statistics (iterations, status) to identify bottlenecks.
"""

import sys
import os
import jax.numpy as jnp
import numpy as np
import time

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.utils.user_types import PlannerData

# Mocking the ellipsoid_cbf to avoid dependency on examples folder
def ellipsoid_cbf(obstacle, ellipsoid):
    """
    Creates a CBF for an ellipsoidal obstacle.
    h(x) = ((x - obs)^T P (x - obs)) - 1 >= 0
    where P = diag(1/a^2, 1/b^2)
    """
    center = obstacle[:2]
    radii = ellipsoid
    P = jnp.diag(1.0 / (radii**2))

    def h(x_and_t):
        pos = x_and_t[:2]
        diff = pos - center
        return jnp.dot(diff, jnp.dot(P, diff)) - 1.0

    return h

def run_benchmark():
    print("🔎 Scout: Starting Solver Statistics Benchmark...")

    # Simulation parameters
    tf = 5.0
    dt = 0.01
    num_steps = int(tf / dt)

    init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
    desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
    actuation_constraints = jnp.array([100.0, 100.0])

    unicycle_dynamics = unicycle.plant(lam=1.0)

    # Simple nominal controller
    uniycle_nom_controller = unicycle.controllers.proportional_controller(
        dynamics=unicycle_dynamics,
        Kp_pos=1.0,
        Kp_theta=1.0,
    )

    # Obstacles
    obstacles = [
        (1.0, 2.0, 0.0),
        (3.0, 2.0, 0.0),
    ]
    ellipsoids = [
        (0.5, 1.5),
        (0.75, 2.0),
    ]

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

    print(f"Running simulation for {num_steps} steps...")
    start_time = time.time()

    # Run with JIT to test JIT extraction path
    results = sim.execute(
        x0=init_state,
        dt=dt,
        num_steps=num_steps,
        dynamics=unicycle_dynamics,
        integrator=integrator,
        nominal_controller=uniycle_nom_controller,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        filepath=None, # No file logging
        verbose=True,
        planner_data=PlannerData(
            x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, num_steps + 1)),
        ),
        use_jit=True,
    )

    end_time = time.time()
    print(f"Simulation complete in {end_time - start_time:.4f} seconds.")

    # Extract solver stats
    controller_data = results.controller_data

    # Debug info
    # print("Keys:", list(controller_data.keys()))

    target_key = "sub_data_solver_iter"

    if target_key in controller_data:
        iters = controller_data[target_key]
        # Convert to numpy for stats
        iters_np = np.array(iters)

        print("\n📊 Solver Statistics:")
        print(f"  Mean Iterations: {np.mean(iters_np):.2f}")
        print(f"  Max Iterations:  {np.max(iters_np)}")
        print(f"  Min Iterations:  {np.min(iters_np)}")
        print(f"  Total Solves:    {len(iters_np)}")

        # Check for high iterations
        high_iters = np.sum(iters_np > 100)
        print(f"  Steps > 100 iters: {high_iters} ({high_iters/len(iters_np)*100:.1f}%)")

    else:
        print(f"\n❌ Error: '{target_key}' not found in controller data.")
        print("Available keys:", list(controller_data.keys()))

if __name__ == "__main__":
    run_benchmark()
