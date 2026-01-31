# Bolt's Journal

## 2025-01-31 - Python Loop Overhead in JIT Simulation
**Learning:** Even with JAX JIT (`jax.lax.scan`), iterating over result arrays in Python to repack them into objects (e.g., `SimulationStepData`) can introduce massive overhead (orders of magnitude) for long horizons.
**Action:** When using JIT, return stacked arrays directly and process them in bulk. Avoid unpacking to Python objects unless absolutely necessary (e.g., for legacy callbacks that can't be vectorized).
