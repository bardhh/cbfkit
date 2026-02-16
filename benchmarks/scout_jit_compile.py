"""
Scout: JAX Compile Counter
Measures JIT recompilation frequency when varying simulation horizon.

Run:
  python benchmarks/scout_jit_compile.py
"""
import time
import jax.numpy as jnp
from jax import random

from cbfkit.simulation import simulator
from cbfkit.utils.jit_monitor import JitMonitor
from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator
from cbfkit.integration.forward_euler import forward_euler

# Create dynamics ONCE to avoid recompilation due to new function object
DYNAMICS = two_dimensional_single_integrator()

def run_sim(num_steps: int) -> float:
    """Runs a minimal simulation and returns execution time."""
    x0 = jnp.array([0.0, 0.0])
    dt = 0.01

    start_time = time.time()

    # Run simulator with JIT enabled
    simulator.execute(
        x0=x0,
        dt=dt,
        num_steps=num_steps,
        dynamics=DYNAMICS,
        integrator=forward_euler,
        use_jit=True,
        verbose=False
    )

    end_time = time.time()
    return end_time - start_time


def main():
    print("🔍 Scout: JAX Compile Diagnostic")
    print("--------------------------------")
    print("Goal: Reveal hidden recompilations when changing 'num_steps'.")
    print("Expectation: Changing 'num_steps' triggers a full JIT recompile.\n")

    # Reset monitor to start clean
    JitMonitor.reset()

    # Define test scenarios: (Description, num_steps)
    scenarios = [
        ("Warmup (N=100)", 100),
        ("Fast (N=100)", 100),
        ("New Horizon (N=101)", 101),
        ("Fast (N=101)", 101),
        ("New Horizon (N=102)", 102),
        ("Revert (N=100)", 100),
    ]

    # Header
    print(f"{'Scenario':<25} | {'Steps':<6} | {'Time (s)':<10} | {'Compiles':<15}")
    print("-" * 65)

    cumulative_compiles = 0

    for name, n in scenarios:
        # Check initial count for simulator_jit
        initial_counts = JitMonitor.get_counts().get("simulator_jit", 0)

        duration = run_sim(n)

        # Check final count
        final_counts = JitMonitor.get_counts().get("simulator_jit", 0)

        delta = final_counts - initial_counts

        # Format compile info: "+1 (Total: 5)" or "0 (Total: 5)"
        # We want to see +0 if it didn't recompile
        if delta > 0:
            compile_str = f"+{delta} (Total: {final_counts})"
        else:
            compile_str = f" 0 (Total: {final_counts})"

        print(f"{name:<25} | {n:<6} | {duration:<10.4f} | {compile_str:<15}")

    print("-" * 65)
    print("\nInterpretation:")
    print(" - 'Time': High values indicate JIT compilation overhead.")
    print(" - 'Compiles': Shows if 'simulator_jit' was recompiled.")
    print(" - Result: Changing 'num_steps' forces recompilation because it is a static argument.")
    print(" - Fix: To avoid recompilation, pad 'num_steps' to a fixed max length or avoid varying it.")


if __name__ == "__main__":
    main()
