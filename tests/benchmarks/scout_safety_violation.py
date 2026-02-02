"""
Scout: Safety Violation & Slack Usage Benchmark
===============================================
Measures the magnitude of safety violations (h < 0) and slack variable usage
when using relaxable CBFs.
Reveals "Silent Failures" where the solver succeeds but safety is compromised.
"""

import sys
import os
import time
import jax.numpy as jnp
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates import rectify_relative_degree, concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller
from cbfkit.integration import runge_kutta_4 as integrator

def ellipsoid_cbf(obstacle, ellipsoid):
    center = obstacle[:2]
    radii = ellipsoid
    P = jnp.diag(1.0 / (radii**2))
    def h(x_and_t):
        pos = x_and_t[:2]
        diff = pos - center
        return jnp.dot(diff, jnp.dot(P, diff)) - 1.0
    return h

def run_scout():
    print("🔎 Scout: Safety Violation & Slack Usage")
    print("---------------------------------------")

    # 1. Setup Scenario: Collision Course
    # Start at (0,0), Goal at (4,0). Obstacle at (2,0).
    dt = 0.02
    tf = 4.0
    steps = int(tf / dt)

    x0 = jnp.array([0.0, 0.0, 0.0, 0.0]) # x, y, v, theta

    # Dynamics
    dyn = unicycle.plant(lam=1.0)
    limits = jnp.array([5.0, 5.0]) # a_max, omega_max

    # Nominal Controller: P controller driving straight to goal
    # This ignores obstacles
    def nominal_controller(t, x, key=None, data=None):
        return jnp.array([2.0, 0.0]), {} # Accelerate hard, no turn

    # Obstacle
    obs_pos = jnp.array([2.0, 0.0, 0.0])
    obs_rad = jnp.array([0.5, 0.5])

    barrier = rectify_relative_degree(
        function=ellipsoid_cbf(obs_pos, obs_rad),
        system_dynamics=dyn,
        state_dim=4,
        form="exponential",
    )(
        certificate_conditions=zeroing_barriers.linear_class_k(1.0),
    )

    # 2. Configure CBF Controller with LOW penalty
    # This simulates a user configuring the solver loosely
    # We expect the robot to "cut the corner" or hit the obstacle
    controller = cbf_controller(
        control_limits=limits,
        dynamics_func=dyn,
        barriers=concatenate_certificates(barrier),
        relaxable_cbf=True,
        slack_penalty_cbf=0.1, # Extremely low penalty!
        slack_bound_cbf=100.0,
    )

    print(f"Running simulation (Penalty={0.1})...")
    results = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=steps,
        dynamics=dyn,
        integrator=integrator,
        nominal_controller=nominal_controller,
        controller=controller,
        verbose=False,
    )

    # 3. Analyze Results
    xs = np.array(results.states)

    # Compute h(x)
    # Re-instantiate h function logic
    P = np.diag(1.0 / (obs_rad**2))
    hs = []
    for x in xs:
        pos = x[:2]
        diff = pos - obs_pos[:2]
        val = np.dot(diff, np.dot(P, diff)) - 1.0
        hs.append(val)
    hs = np.array(hs)

    # Extract Slacks
    # cbf_clf_qp_generator output: u (n_con), slack_cbf (n_bfs), slack_clf (n_lfs)
    # n_con=2, n_bfs=1. Slack is at index 2.
    # Note: sim.execute results.controller_data_values contains 'sol' if available

    slacks = []
    try:
        data = results.controller_data
        if "sol" in data:
            sols = data["sol"]
            # sols is (N, n_vars)
            if sols is not None:
                 # Check shapes. If using JIT, might be stacked.
                 # sim.execute returns JAX arrays in results
                 sols = np.array(sols)
                 # Index 2 is slack (u0, u1, delta)
                 if sols.shape[1] > 2:
                     slacks = sols[:, 2]
    except Exception as e:
        print(f"Error extracting slack: {e}")
        # Fallback debug
        print(f"Results type: {type(results)}")
        print(f"Results keys: {results._fields if hasattr(results, '_fields') else 'N/A'}")

    slacks = np.array(slacks)

    # 4. Report
    min_h = np.min(hs)
    max_slack = np.max(slacks) if len(slacks) > 0 else 0.0
    mean_slack = np.mean(slacks) if len(slacks) > 0 else 0.0

    print("\n📊 Safety & Slack Metrics")
    print(f"  Min Barrier Value h(x): {min_h:.4f} " + ("❌ VIOLATION" if min_h < 0 else "✅ SAFE"))
    print(f"  Max Slack Used:         {max_slack:.4f}")
    print(f"  Mean Slack Used:        {mean_slack:.4f}")

    if min_h < -0.01:
        print("\n⚠️  Severe Violation Detected!")
        print("   The solver relaxed the constraint significantly.")
        print("   Check 'slack_penalty_cbf' or 'relaxable_cbf'.")
    elif min_h < 0:
        print("\n⚠️  Minor Violation.")
    else:
        print("\n✅ No Violation.")

    if max_slack > 1.0:
        print(f"⚠️  High slack usage ({max_slack:.2f}).")

if __name__ == "__main__":
    run_scout()
