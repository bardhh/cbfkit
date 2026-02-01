## 2024-05-22 - [JAX Shape Mismatch]
Learning: JAX broadcasting rules can silently turn (N, 1) vs (N,) operations into (N, N) matrices, causing hidden overhead. `jaxopt` handles both, but keeping vectors 1D avoids ambiguity and reshape/expand_dims calls.
Action: Prefer 1D arrays for vectors (h_vec, q_vec) in QP formulations unless matrix operations explicitly require column vectors.

## 2026-02-01 - [Avoid Post-Hoc Array Updates]
**Learning:** Updating large JAX arrays (e.g., `g_mat_c.at[...].multiply(scale)`) inside a JIT-compiled loop incurs unnecessary allocation and copy overhead.
**Action:** Pass scalar parameters (like `scale`) into the generator functions (via `kwargs`) so the values are computed correctly during construction, avoiding the need for updates.

## 2026-03-08 - [Optimizing Diagonal QP Costs]
Learning: Using `jnp.diag(v)` creates a dense matrix, making `P @ x` cost `O(N^2)`. For diagonal cost matrices, element-wise `v * x` is `O(N)` and avoids large matrix materialization.
Action: Pre-compute diagonal vectors and use element-wise multiplication in QP cost generation whenever `P` structure is known (e.g., `auto_p_mat`).

## 2026-03-08 - [Single-Pass Constraint Stacking]
Learning: Nested `vstack` calls (e.g., `vstack(u, vstack(c, l))`) create intermediate arrays that increase memory churn. Flattening the hierarchy to `vstack(u, c, l)` allows allocation in one go.
Action: Inline constraint generation logic if it returns stacked arrays, or refactor to return components, to enable single-pass stacking at the consumer level.
