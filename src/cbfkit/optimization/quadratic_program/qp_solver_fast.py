"""Backward-compatible shim — coord descent was replaced by PDIPM.

The original dual coordinate descent implementation could not converge on
slack-relaxed CBF-CLF-QPs (verified failure on mppi_cbf_reach_avoid step 0).
See docs/superpowers/specs/2026-05-26-fast-qp-pdipm-design.md for the rationale.

This module re-exports ``solve_qp_fast`` so existing imports keep working;
internally it just calls ``solve_qp_pdipm`` from ``qp_solver_pdipm``.
"""

from typing import Optional, Tuple

from jax import Array

from cbfkit.optimization.quadratic_program.qp_solver_pdipm import (
    PdipmState,
    solve_qp_pdipm,
)


def solve_qp_fast(
    P: Array,
    q: Array,
    G: Array,
    h: Array,
    warm_start: Optional[Array] = None,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> Tuple[Array, int, Array]:
    """Backward-compatible alias for solve_qp_pdipm.

    Returns ``(x, status, dual)``. For full state including primal slacks
    (needed for richer warm-starting), call ``solve_qp_pdipm`` directly.

    Note: the legacy interface accepted an Array as ``warm_start`` (the dual
    only). To preserve that signature, we wrap it in a minimal ``PdipmState``
    with slacks defaulting to ones, which the PDIPM init will then clamp
    into the interior.
    """
    warm = None
    if warm_start is not None:
        # Best-effort reconstruction of state from legacy dual-only seed.
        # Real warm-starting should pass a full PdipmState via solve_qp_pdipm.
        import jax.numpy as jnp

        warm = PdipmState(
            x=jnp.zeros(P.shape[0]),
            s=jnp.ones(G.shape[0]),
            dual=warm_start,
            iter_num=0,
        )
    x, status, state = solve_qp_pdipm(
        P,
        q,
        G,
        h,
        warm_start=warm,
        max_iter=max_iter,
        tol=tol,
    )
    return x, status, state.dual
