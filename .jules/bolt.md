## 2024-05-22 - [JAX Shape Mismatch]
Learning: JAX broadcasting rules can silently turn (N, 1) vs (N,) operations into (N, N) matrices, causing hidden overhead. `jaxopt` handles both, but keeping vectors 1D avoids ambiguity and reshape/expand_dims calls.
Action: Prefer 1D arrays for vectors (h_vec, q_vec) in QP formulations unless matrix operations explicitly require column vectors.

## 2026-02-01 - [Avoid Post-Hoc Array Updates]
**Learning:** Updating large JAX arrays (e.g., `g_mat_c.at[...].multiply(scale)`) inside a JIT-compiled loop incurs unnecessary allocation and copy overhead.
**Action:** Pass scalar parameters (like `scale`) into the generator functions (via `kwargs`) so the values are computed correctly during construction, avoiding the need for updates.
