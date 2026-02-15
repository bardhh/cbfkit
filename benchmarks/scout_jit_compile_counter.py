"""Scout Benchmark: JIT Compilation Counter"""
import time
import jax.numpy as jnp
from cbfkit.simulation import simulator as sim
from cbfkit.systems.unicycle.models.accel_unicycle import plant
from cbfkit.integration import forward_euler
from cbfkit.utils.jit_monitor import JitMonitor

def run():
    JitMonitor.reset()
    dyn, x0, dt = plant(), jnp.zeros(4), 0.01
    scenarios = [("Base", 100), ("Repeat", 100), ("Mod", 101)]

    print(f"{'Run':<10} | {'N':<3} | {'Time':<6} | {'JIT':<3} | {'Status'}")
    prev_jit = 0

    for name, n in scenarios:
        t0 = time.time()
        sim.execute(x0, dt, n, dyn, forward_euler, use_jit=True, verbose=False)
        elap = time.time() - t0
        curr_jit = JitMonitor.get_counts().get("simulator_jit", 0)

        stat = "Recompile" if curr_jit > prev_jit else ("Cached" if prev_jit > 0 else "Unknown")
        print(f"{name:<10} | {n:<3} | {elap:.4f} | {curr_jit:<3} | {stat}")
        prev_jit = curr_jit

if __name__ == "__main__":
    run()
