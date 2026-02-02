"""
Scout Benchmark: Runtime Breakdown
==================================

Measures the per-step runtime breakdown of a simulation, separating:
- Dynamics Integration (Physics)
- Constraint Generation (Dynamics calls inside Controller)
- Solver & Overhead (Controller internal logic)

Run with:
    python tests/benchmarks/scout_runtime_breakdown.py
"""

import sys
import os
import time
import jax
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

class ProbedFunction:
    """Wraps a callable to measure execution time, ensuring JAX sync."""
    def __init__(self, func, name):
        self.func = func
        self.name = name
        self.times = []
        self.calls = 0

    def __call__(self, *args, **kwargs):
        # We start timing BEFORE dispatch
        t0 = time.perf_counter()

        out = self.func(*args, **kwargs)

        # We must block to measure actual execution time on device (or even CPU async)
        self._block(out)

        t1 = time.perf_counter()
        self.times.append((t1 - t0) * 1e3) # Convert to ms
        self.calls += 1
        return out

    def _block(self, obj):
        """Recursively block until ready for JAX arrays."""
        if hasattr(obj, 'block_until_ready'):
            obj.block_until_ready()
        elif isinstance(obj, (tuple, list)):
            for x in obj:
                self._block(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                self._block(x)

    def stats(self):
        arr = np.array(self.times)
        if len(arr) == 0:
            return 0.0, 0.0, 0
        return np.mean(arr), np.sum(arr), self.calls

def run_benchmark():
    print("🔎 Scout: Runtime Breakdown (Python Loop)")
    print("---------------------------------------")

    # 1. Setup System
    # Use unicycle model
    raw_dynamics = unicycle.plant(lam=1.0)

    # Create two probes: one for simulator integrator, one for controller constraints
    dyn_sim = ProbedFunction(raw_dynamics, "Dynamics (Integration)")
    dyn_ctl = ProbedFunction(raw_dynamics, "Dynamics (Constraints)")

    dt = 0.01
    tf = 1.0 # Short run is enough for profiling
    steps = int(tf / dt)
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])

    # 2. Setup Controller
    # We pass dyn_ctl to the barrier and controller so we measure those calls separate from integrator

    # Obstacle
    obs = jnp.array([2.0, 0.0, 0.0])
    ell = jnp.array([0.5, 0.5])

    def ellipsoid_cbf(obstacle, ellipsoid):
        P = jnp.diag(1.0 / (ellipsoid**2))
        def h(x_and_t):
            pos = x_and_t[:2]
            diff = pos - obstacle[:2]
            return jnp.dot(diff, jnp.dot(P, diff)) - 1.0
        return h

    barrier = rectify_relative_degree(
        function=ellipsoid_cbf(obs, ell),
        system_dynamics=dyn_ctl, # Probe here!
        state_dim=4,
        form="exponential",
    )(
        certificate_conditions=zeroing_barriers.linear_class_k(1.0),
    )

    raw_controller = cbf_controller(
        control_limits=jnp.array([10.0, 10.0]),
        dynamics_func=dyn_ctl, # Probe here!
        barriers=concatenate_certificates(barrier),
    )

    # Probe the top-level controller
    ctl_probed = ProbedFunction(raw_controller, "Controller (Total)")

    print(f"Running simulation for {steps} steps (use_jit=False)...")
    print("  Note: Using jax.disable_jit() to force op-by-op execution for profiling.")
    print("        This adds dispatch overhead but reveals the logic breakdown.")

    # Resetting probes (just in case)
    dyn_sim.times = []
    dyn_sim.calls = 0
    dyn_ctl.times = []
    dyn_ctl.calls = 0
    ctl_probed.times = []
    ctl_probed.calls = 0

    with jax.disable_jit():
        sim.execute(
            x0=x0,
            dt=dt,
            num_steps=steps,
            dynamics=dyn_sim, # Probe here!
            integrator=integrator,
            controller=ctl_probed, # Probe here!
            verbose=False,
            use_jit=False, # Essential for per-call instrumentation
        )

    # 3. Report
    mean_sim, sum_sim, n_sim = dyn_sim.stats()
    mean_ctl_dyn, sum_ctl_dyn, n_ctl_dyn = dyn_ctl.stats()
    mean_ctl, sum_ctl, n_ctl = ctl_probed.stats()

    # Derived stats
    # Controller Time = Constraint Gen (Dyn) + Solver/Overhead
    # So Solver/Overhead = Controller - Constraint Gen (Dyn)
    mean_solver = mean_ctl - mean_ctl_dyn

    print("\n⏱️  Runtime Breakdown (Mean per Step)")
    print(f"  {'Component':<25} {'Time (ms)':<10} {'Calls/Step':<10} {'% of Loop'}")

    # Note: Simulator loop calls dynamics 4 times per step (RK4) usually
    # and controller once.

    calls_per_step_sim = n_sim / steps
    calls_per_step_ctl_dyn = n_ctl_dyn / steps

    # Approximate total loop time (sum of components)
    # Note: Integration has overhead too, but dynamics is the bulk.
    total_loop_time = (mean_sim * calls_per_step_sim) + mean_ctl

    row_fmt = "  {:<25} {:<10.3f} {:<10.1f} {:<5.1f}%"

    print("-" * 60)
    print(row_fmt.format("Dynamics (Integration)", mean_sim * calls_per_step_sim, calls_per_step_sim, (mean_sim * calls_per_step_sim)/total_loop_time*100))
    print(row_fmt.format("Dynamics (Constraints)", mean_ctl_dyn * calls_per_step_ctl_dyn, calls_per_step_ctl_dyn, (mean_ctl_dyn * calls_per_step_ctl_dyn)/total_loop_time*100))
    print(row_fmt.format("Solver & Logic", mean_solver, 1.0, (mean_solver)/total_loop_time*100))
    print("-" * 60)
    print(f"  {'Total Est. Step Time':<25} {total_loop_time:<10.3f}")

    # Interpretation
    print("\n💡 Interpretation:")
    if mean_solver > 5 * (mean_sim + mean_ctl_dyn):
         print("  Solver dominates runtime. Consider:")
         print("  - Reducing barrier count or horizon.")
         print("  - Tuning solver settings (tol, max_iter).")
         print("  - Using JIT (use_jit=True) to fuse operations.")
    elif mean_sim > mean_solver:
         print("  Dynamics integration is expensive. Consider:")
         print("  - Simpler dynamics model.")
         print("  - Lower order integrator (Euler vs RK4).")
    else:
         print("  Balanced workload.")

if __name__ == "__main__":
    run_benchmark()
