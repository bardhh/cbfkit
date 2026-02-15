"""
Scout Diagnostic: JIT Recompilation Check.

This script verifies that the JIT compiler is behaving as expected:
1. Compiling once for the first run.
2. Reusing the compiled function for subsequent runs with identical static arguments.
3. Recompiling when static arguments change (e.g. dynamics function).

Command to run:
    python examples/diagnostics/jit_recompile_check.py
"""

import jax
import jax.numpy as jnp
from cbfkit.simulation import simulator
from cbfkit.integration import forward_euler
from cbfkit.utils.jit_monitor import JitMonitor

def run_simulation(dynamics_func):
    """Runs a minimal simulation."""
    x0 = jnp.array([0.0, 0.0])
    dt = 0.01
    num_steps = 10

    simulator.execute(
        x0=x0,
        dt=dt,
        num_steps=num_steps,
        dynamics=dynamics_func,
        integrator=forward_euler,
        use_jit=True,
        verbose=False
    )

def main():
    print("🔍 Scout: Checking JIT compilation behavior...")

    # Reset monitor
    JitMonitor.reset()

    # ---------------------------------------------------------
    # Scenario 1: First Run
    # ---------------------------------------------------------
    def dynamics_v1(x):
        return jnp.zeros_like(x), jnp.zeros((2, 1))

    print("\n[1] Running Simulation 1 (First Call)...")
    run_simulation(dynamics_v1)

    counts = JitMonitor.get_counts()
    sim_jit_count = counts.get("simulator_jit", 0)
    print(f"    -> Compilation Counts: {counts}")

    if sim_jit_count != 1:
        print(f"❌ FAILED: Expected 1 compilation, got {sim_jit_count}")
        exit(1)

    # ---------------------------------------------------------
    # Scenario 2: Second Run (Identical)
    # ---------------------------------------------------------
    print("\n[2] Running Simulation 2 (Identical Call)...")
    run_simulation(dynamics_v1)

    counts = JitMonitor.get_counts()
    sim_jit_count_2 = counts.get("simulator_jit", 0)
    print(f"    -> Compilation Counts: {counts}")

    if sim_jit_count_2 != 1:
        print(f"❌ FAILED: Expected count to remain 1, got {sim_jit_count_2}. Redundant recompilation detected!")
        exit(1)

    # ---------------------------------------------------------
    # Scenario 3: Third Run (Changed Dynamics - should recompile)
    # ---------------------------------------------------------
    print("\n[3] Running Simulation 3 (New Dynamics Function)...")

    # Even if code is same, a new function object is a new static arg
    def dynamics_v2(x):
        return jnp.zeros_like(x), jnp.zeros((2, 1))

    run_simulation(dynamics_v2)

    counts = JitMonitor.get_counts()
    sim_jit_count_3 = counts.get("simulator_jit", 0)
    print(f"    -> Compilation Counts: {counts}")

    if sim_jit_count_3 != 2:
        print(f"❌ FAILED: Expected count to increase to 2, got {sim_jit_count_3}. Failed to detect new function.")
        exit(1)

    print("\n✅ SUCCESS: JIT compilation monitoring is working correctly.")
    print("   Run this tool to detect redundant recompilations in your loops.")

if __name__ == "__main__":
    main()
