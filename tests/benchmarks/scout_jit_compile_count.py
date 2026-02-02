"""
Scout: JIT Compile Count Benchmark
==================================

Measures the number of JIT compilations triggered by a function.
Excessive recompilation is a major performance bottleneck in JAX.

This benchmark demonstrates:
1. Ideal case: Constant cache size (1) for stable inputs.
2. Pathological case: Growing cache size for changing input shapes/types.
"""

import sys
import os
import time
import jax
import jax.numpy as jnp
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

def run():
    print("🔎 Scout: Starting JIT Compile Count Benchmark...")

    # Define a simple compute heavy function
    @jax.jit
    def heavy_compute(x, param):
        # Simulate some controller logic
        return jnp.dot(x, x.T) * param

    print("\n1. Stable Inputs (Ideal)")
    print("   Running 100 steps with fixed shape (10,)...")

    for _ in range(100):
        x = jnp.ones((10,))
        heavy_compute(x, 1.0)

    cache_size = heavy_compute._cache_size()
    print(f"   Cache Size: {cache_size} (Expected: 1)")
    if cache_size == 1:
        print("   ✅ GOOD: No recompilation.")
    else:
        print(f"   ❌ BAD: {cache_size} compilations detected.")

    print("\n2. Variable Shapes (Pathological)")
    print("   Running 20 steps with changing shapes...")

    # We expect this to trigger recompilation for every unique shape
    # This simulates e.g. a variable number of obstacles or agents not being padded

    for i in range(1, 21):
        x = jnp.ones((i,)) # Shape changes: (1,), (2,), ...
        heavy_compute(x, 1.0)

    cache_size = heavy_compute._cache_size()
    print(f"   Cache Size: {cache_size} (Expected: ~21)")

    if cache_size > 10:
        print("   ⚠️  WARNING: High recompilation count detected!")
        print("      This happens when input shapes or static arguments change.")
        print("      Fix: Pad inputs to fixed size or use `jax.lax.cond`.")

    print("\n3. Scalar Types (Subtle Trap)")
    # Python int vs float sometimes triggers recompile if not careful

    @jax.jit
    def type_sensitive(val):
        return val * 2.0

    type_sensitive(1)   # int
    type_sensitive(1.0) # float

    print(f"   Mixed int/float calls. Cache Size: {type_sensitive._cache_size()}")

    print("\nDone.")

if __name__ == "__main__":
    run()
