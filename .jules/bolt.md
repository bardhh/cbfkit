# Bolt's Journal

## 2025-01-31 - Python Loop Overhead in JIT Simulation
**Learning:** Even with JAX JIT (`jax.lax.scan`), iterating over result arrays in Python to repack them into objects (e.g., `SimulationStepData`) can introduce massive overhead (orders of magnitude) for long horizons.
**Action:** When using JIT, return stacked arrays directly and process them in bulk. Avoid unpacking to Python objects unless absolutely necessary (e.g., for legacy callbacks that can't be vectorized).

## 2025-02-14 - Optimized Data Formatting in Simulation
**Learning:** Formatting simulation results by iterating over a tuple of NamedTuples and rebuilding arrays (`jnp.array([step.field for step in data])`) is slow for large N.
**Action:** Transpose the tuple of NamedTuples (`SimulationStepData(*zip(*data))`) and use `jnp.stack` on the resulting columns. This reduces iteration overhead and leverages JAX's stacking efficiency.

## 2025-02-19 - Repeated Graph Construction in JIT Simulator
**Learning:** `simulator_jit` (which uses `lax.scan`) was not itself JIT-compiled. This meant that every call to `execute(use_jit=True)` re-executed the Python logic to build the `scan` graph. While `lax.scan` caches the compiled kernel, the graph construction overhead (Python side) was significant (~240ms per call).
**Action:** Decorated `simulator_jit` with `@partial(jax.jit, static_argnames=[...])` to JIT-compile the graph construction itself, eliminating Python overhead on subsequent calls.

## 2025-02-20 - JAX CSE Effectiveness and Logging Bottlenecks
**Learning:** Manual Common Subexpression Elimination (CSE) of `dynamics(x)` calls inside `scan_step` yielded negligible speedup (<1%), confirming that JAX XLA is highly effective at optimizing pure function calls. However, the host-side logging loop in `simulator.py` (converting JAX arrays to Python dicts step-by-step) was a 2.7x performance drag.
**Action:** Trust XLA for graph optimizations. Focus on eliminating Python loops in data transfer paths (logging, plotting) by using vectorized/bulk operations.

## 2025-02-21 - lax.scan vs lax.fori_loop for Rollouts
**Learning:** Using `lax.fori_loop` with `array.at[i].set(...)` for sequential rollouts creates significant overhead compared to `lax.scan`, even when JIT-compiled. In MPPI rollouts, switching to `lax.scan` yielded a ~7x speedup (247ms -> 36ms).
**Action:** Prefer `lax.scan` for sequential accumulation or dynamics rollouts. It handles memory access more efficiently and avoids explicit buffer updates.

## 2025-10-26 - Redundant Dynamics Evaluation in Python Loop
**Learning:** The Python-based `stepper` (used for non-JIT simulation) evaluated system dynamics twice per step when using `forward_euler`: once for logging/controller and once inside the integrator.
**Action:** Special-cased `forward_euler` in the stepper to reuse the already computed dynamics, yielding a ~12% speedup in simulations with moderate dynamics complexity.
