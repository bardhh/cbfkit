## 2024-05-22 - [JAX Shape Mismatch]
Learning: JAX broadcasting rules can silently turn (N, 1) vs (N,) operations into (N, N) matrices, causing hidden overhead. `jaxopt` handles both, but keeping vectors 1D avoids ambiguity and reshape/expand_dims calls.
Action: Prefer 1D arrays for vectors (h_vec, q_vec) in QP formulations unless matrix operations explicitly require column vectors.
