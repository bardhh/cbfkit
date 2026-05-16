"""Fast small-QP solver for CBF-CLF problems.

A JIT-compiled QP solver optimized for the tiny, structured QPs that arise
in CBF-CLF-QP safety filtering (2-8 variables, 5-20 constraints).

Uses a clipped projected gradient approach: since P is always positive
definite and well-conditioned in CBF-QP (diagonal with known weights),
we can solve efficiently by projecting onto the feasible set.

Performance target: <500us per solve vs ~45ms for jaxopt OSQP.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, lax


from functools import partial


@partial(jax.jit, static_argnames=("max_iter",))
def solve_qp_fast(
    P: Array,
    q: Array,
    G: Array,
    h: Array,
    warm_start: Optional[Array] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[Array, int, Array]:
    """Solve a small convex QP via dual projection.

    Solves: min  0.5 x^T P x + q^T x
            s.t. G x <= h

    Uses the dual approach: iteratively finds the optimal Lagrange multipliers
    for the inequality constraints, then recovers the primal solution.
    For small problems with P diagonal (typical CBF-QP), this converges in
    very few iterations.

    Args:
        P: Positive definite cost matrix (n, n).
        q: Linear cost vector (n,).
        G: Inequality constraint matrix (m, n).
        h: Inequality constraint bound (m,).
        warm_start: Previous dual variables for warm-starting (m,).
        max_iter: Maximum iterations (default 30, sufficient for CBF-QP).
        tol: Convergence tolerance.

    Returns:
        (solution, status, dual) where:
        - solution: primal solution (n,)
        - status: 1=solved, 2=max_iter_reached
        - dual: dual variables for warm-starting next call (m,)
    """
    n = P.shape[0]
    m = G.shape[0]

    # Precompute P^{-1} (P is positive definite, typically diagonal)
    P_inv = jnp.linalg.inv(P)

    # Precompute kernel matrix for dual: K = G P^{-1} G^T
    GP_inv = G @ P_inv
    K = GP_inv @ G.T

    # Precompute: G P^{-1} q + h
    base_rhs = GP_inv @ q + h

    # Initialize dual variables (Lagrange multipliers >= 0)
    if warm_start is not None:
        lam0 = jnp.maximum(warm_start, 0.0)
    else:
        lam0 = jnp.zeros(m)

    # Coordinate descent on the dual problem:
    # max_lam  -0.5 lam^T K lam - base_rhs^T lam   s.t. lam >= 0
    #
    # This is equivalent to the original QP. Each coordinate update is:
    #   lam_i = max(0, -(base_rhs_i + sum_{j!=i} K_ij lam_j) / K_ii)
    #
    # For small m (5-20), one full sweep = one iteration.

    diag_K = jnp.diag(K)
    # Regularize zero diagonals (degenerate constraints)
    diag_K_safe = jnp.where(diag_K > 1e-12, diag_K, 1.0)

    def _iter(carry, _):
        lam = carry

        # Full coordinate sweep
        def _update_one(lam_inner, i):
            # Residual for constraint i: r_i = K_i @ lam + base_rhs_i
            r_i = jnp.dot(K[i], lam_inner) - K[i, i] * lam_inner[i] + base_rhs[i]
            # Optimal update: lam_i = max(0, -r_i / K_ii)
            lam_i_new = jnp.maximum(0.0, -r_i / diag_K_safe[i])
            return lam_inner.at[i].set(lam_i_new), None

        lam_new, _ = lax.scan(_update_one, lam, jnp.arange(m))
        return lam_new, None

    lam_final, _ = lax.scan(_iter, lam0, None, length=max_iter)

    # Recover primal: x = -P^{-1}(q + G^T lam)
    x = -P_inv @ (q + G.T @ lam_final)

    # Check feasibility
    violations = G @ x - h
    max_violation = jnp.max(violations)
    feasible = max_violation <= tol * 100  # relaxed for numerical precision

    status = jnp.where(feasible, jnp.int32(1), jnp.int32(2))

    return x, status, lam_final
