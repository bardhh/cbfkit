
## 2025-05-22 - JAX/OSQP Diagonal P Pitfall
**Learning:** When using `jaxopt.OSQP`, passing a 1D vector for the `P` matrix (to represent a diagonal matrix) triggers a fallback to a slow iterative solver (CG/GMRES) instead of using the efficient dense Cholesky factorization path used when `P` is a 2D dense array. This caused an 8x slowdown in simulations.
**Action:** Always pass `p_mat` as a dense 2D array (e.g., `jnp.diag(vec)`) to `jaxopt.OSQP` for small/dense problems, even if the math allows diagonal structure. Optimize `q_vec` calculation separately if needed.
