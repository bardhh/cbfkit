# Bolt's Journal

## 2025-01-31 - Python Loop Overhead in JIT Simulation
**Learning:** Even with JAX JIT (`jax.lax.scan`), iterating over result arrays in Python to repack them into objects (e.g., `SimulationStepData`) can introduce massive overhead (orders of magnitude) for long horizons.
**Action:** When using JIT, return stacked arrays directly and process them in bulk. Avoid unpacking to Python objects unless absolutely necessary (e.g., for legacy callbacks that can't be vectorized).

## 2025-02-14 - Optimized Data Formatting in Simulation
**Learning:** Formatting simulation results by iterating over a tuple of NamedTuples and rebuilding arrays (`jnp.array([step.field for step in data])`) is slow for large N.
**Action:** Transpose the tuple of NamedTuples (`SimulationStepData(*zip(*data))`) and use `jnp.stack` on the resulting columns. This reduces iteration overhead and leverages JAX's stacking efficiency.
