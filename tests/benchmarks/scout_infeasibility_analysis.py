"""
Scout: Infeasibility Analysis Benchmark
=======================================

Stress-tests the QP solver by forcing a high-speed approach to an obstacle
with limited control authority, measuring the frequency of failure modes.
"""

import sys, os, jax.numpy as jnp, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cbfkit.systems.unicycle.models import accel_unicycle
from cbfkit.certificates import rectify_relative_degree, concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as make_ctl
from cbfkit.simulation import simulator
from cbfkit.integration import forward_euler

STATUS_MAP = {
    1: "SOLVED", 2: "MAX_ITER", 3: "PRIMAL_INF", 4: "DUAL_INF",
    5: "MAX_ITER_UN", -1: "NAN_DETECT", -2: "NAN_INPUT"
}

def run():
    print("🔎 Scout Infeasibility Analysis")
    # 1. Impossible Scenario: 10m/s towards wall, need 10m to stop, have 4m.
    x0, obs = jnp.array([-5.0, 0.0, 10.0, 0.0]), jnp.array([0.0, 0.0, 1.0]) # x, y, r
    dyn = accel_unicycle.plant()

    # 2. Hard Constraint (relaxable_cbf=False is default for vanilla controller if not specified,
    #    but we ensure strictness by not adding slack vars in config if we were building manually.
    #    The vanilla_cbf_clf_qp_controller wrapper defaults relaxable_cbf=False).
    h = lambda x: (x[0]-obs[0])**2 + (x[1]-obs[1])**2 - obs[2]**2
    barrier = rectify_relative_degree(h, dyn, 4, form="exponential")(
        zeroing_barriers.linear_class_k(10.0)
    )

    # Tight limits: [5.0, 5.0]
    ctl = make_ctl(jnp.array([5.0, 5.0]), dyn, concatenate_certificates(barrier))

    # 3. Run Simulation
    # We expect failure around t=0.4s
    results = simulator.execute(
        x0=x0, dt=0.01, num_steps=100, dynamics=dyn, integrator=forward_euler,
        controller=ctl, use_jit=True
    )

    # 4. Analyze Status
    # Note: Simulator flattens nested dicts in sub_data, so keys are joined with underscores.
    c_data = results.controller_data
    statuses = np.array(c_data["sub_data_solver_status"])
    iters = np.array(c_data["sub_data_solver_iter"])

    print(f"\n{'Status':<12} {'Count':<8} {'Freq %':<8} {'Avg Iter':<8}")
    print("-" * 40)
    for code in np.unique(statuses):
        mask = statuses == code
        name = STATUS_MAP.get(code, f"UNKNOWN({code})")
        print(f"{name:<12} {np.sum(mask):<8} {np.mean(mask)*100:<8.1f} {np.mean(iters[mask]):<8.1f}")

    if np.any(statuses != 1):
        print("\n✅ Validated: Solver failures detected as expected.")
    else:
        print("\n⚠️ Warning: Solver succeeded? Scenario might not be impossible enough.")

if __name__ == "__main__": run()
