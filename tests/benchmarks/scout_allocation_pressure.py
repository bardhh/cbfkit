"""
Scout: Allocation Pressure Benchmark
====================================

Measures Python memory allocation (RAM) during JIT compilation and execution.
High allocation pressure triggers frequent Garbage Collection (GC), causing stutter.

Run with:
    python tests/benchmarks/scout_allocation_pressure.py
"""

import sys, os, tracemalloc, jax, jax.numpy as jnp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cbfkit.systems.unicycle.models import accel_unicycle
from cbfkit.certificates import rectify_relative_degree, concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as make_ctl
from cbfkit.simulation import simulator
from cbfkit.integration import forward_euler
from cbfkit.utils.user_types import ControllerData

def measure(name, func):
    tracemalloc.start()
    func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{name:<20} | Peak: {peak/1024**2:<6.2f} MB | Curr: {current/1024**2:<6.2f} MB")

def run():
    print(f"🔎 Scout: Allocation Pressure (PID: {os.getpid()})")
    print(f"{'Phase':<20} | {'Peak RAM':<10} | {'Net RAM':<10}")
    print("-" * 45)

    dyn = accel_unicycle.plant()
    h = lambda x: x[0] - 2.0 # Simple barrier
    barrier = rectify_relative_degree(h, dyn, 4, form="exponential")(
        zeroing_barriers.linear_class_k(1.0)
    )
    ctl = make_ctl(jnp.array([10., 10.]), dyn, concatenate_certificates(barrier))
    x0 = jnp.zeros(4)

    # 1. JIT Compilation
    def compile_step():
        # Force compilation
        jax.jit(ctl)(0.0, x0, jnp.zeros(2), jax.random.PRNGKey(0), ControllerData())[0].block_until_ready()

    measure("JIT Compilation", compile_step)

    # 2. Python Loop (High Churn)
    def python_loop():
        simulator.execute(
            x0, 0.01, 100, dyn, forward_euler, controller=ctl, use_jit=False, verbose=False
        )

    measure("Python Loop (100)", python_loop)

    # 3. JIT Execution (Low Churn)
    def jit_loop():
        simulator.execute(
            x0, 0.01, 100, dyn, forward_euler, controller=ctl, use_jit=True, verbose=False
        )

    # Warmup JIT loop
    jit_loop()
    measure("JIT Loop (100)", jit_loop)

if __name__ == "__main__": run()
