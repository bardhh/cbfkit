"""Mehrotra predictor-corrector PDIPM for small dense convex QPs.

Solves: min 0.5 x^T P x + q^T x  s.t.  G x <= h

Designed for the QP shapes that arise in CBF-CLF-QP safety filtering
(n ~ 2-8 variables, m ~ 5-30 constraints). The barrier-regularized Newton
system is always well-conditioned, which handles slack-relaxation
ill-conditioning that breaks dual coordinate descent.
"""

from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, lax


class PdipmState(NamedTuple):
    """Solver state. Used for warm-start and exposed via QpSolution.params.

    Field name 'dual' matches the access pattern in
    cbfkit.optimization.quadratic_program.solver_registry (state.dual).
    """

    x: Array  # (n,) primal
    s: Array  # (m,) slacks
    dual: Array  # (m,) Lagrange multipliers (lambda)
    iter_num: int  # number of iterations actually performed


def _step_to_boundary(s: Array, ds: Array) -> Array:
    """Largest alpha in (0, 1] such that s + alpha * ds > 0 elementwise.

    For coordinates where ds >= 0, no constraint (returns inf placeholder).
    For coordinates where ds < 0, the cap is -s / ds.
    """
    ratios = jnp.where(ds < 0, -s / ds, jnp.inf)
    return jnp.minimum(1.0, jnp.min(ratios))


_EPS_PD = 1e-10  # PD regularization on H before Cholesky


def _solve_newton_reduced(P: Array, G: Array, s: Array, lam: Array, rhs: Array) -> Array:
    """Solve (P + G^T diag(lam/s) G + eps*I) dx = rhs via Cholesky.

    The eps*I term guards against rounding-induced non-PD near the optimum.
    """
    n = P.shape[0]
    D = lam / s
    H = P + G.T @ (D[:, None] * G) + _EPS_PD * jnp.eye(n)
    L = jnp.linalg.cholesky(H)
    # dx = H^-1 rhs via two triangular solves
    y = jax.scipy.linalg.solve_triangular(L, rhs, lower=True)
    dx = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
    return dx


_STEP_SAFETY = 0.995  # fraction of step-to-boundary; keeps strict interior


def _pdipm_iteration(
    P: Array,
    q: Array,
    G: Array,
    h: Array,
    x: Array,
    s: Array,
    lam: Array,
) -> Tuple[Array, Array, Array]:
    """One Mehrotra predictor-corrector PDIPM step.

    Returns updated (x, s, lam) all strictly positive in s, lam.
    """
    m = G.shape[0]

    # --- Residuals (predictor: r_c with mu = 0) ---
    r_d = P @ x + G.T @ lam + q
    r_p = G @ x + s - h
    r_c_aff = s * lam

    # --- Predictor (affine) step ---
    # Reduced Newton RHS after eliminating ds, dlam from the KKT system:
    # (P + G^T diag(lam/s) G) dx = -r_d + G^T lam - G^T diag(lam/s) r_p
    #                            = -(r_d + G^T ((lam*r_p - s*lam) / s))
    # which simplifies via r_c_aff = s*lam to -(r_d + G^T ((lam*r_p - r_c_aff)/s)).
    rhs_aff = -(r_d + G.T @ ((lam * r_p - r_c_aff) / s))
    dx_aff = _solve_newton_reduced(P, G, s, lam, rhs_aff)
    ds_aff = -r_p - G @ dx_aff
    dlam_aff = -(r_c_aff + lam * ds_aff) / s

    alpha_p_aff = _step_to_boundary(s, ds_aff)
    alpha_d_aff = _step_to_boundary(lam, dlam_aff)
    # Undamped affine step is what textbook Mehrotra uses to compute mu_aff/sigma.
    # The _STEP_SAFETY damping is reserved for the final corrector update below.
    alpha_aff_raw = jnp.minimum(alpha_p_aff, alpha_d_aff)

    # --- Centering (Mehrotra sigma) ---
    mu = jnp.sum(s * lam) / m
    s_aff_pt = s + alpha_aff_raw * ds_aff
    lam_aff_pt = lam + alpha_aff_raw * dlam_aff
    mu_aff = jnp.sum(s_aff_pt * lam_aff_pt) / m
    sigma = (mu_aff / jnp.maximum(mu, 1e-30)) ** 3

    # --- Corrector ---
    r_c = s * lam + ds_aff * dlam_aff - sigma * mu
    rhs = -(r_d + G.T @ ((lam * r_p - r_c) / s))
    dx = _solve_newton_reduced(P, G, s, lam, rhs)
    ds = -r_p - G @ dx
    dlam = -(r_c + lam * ds) / s

    # --- Step ---
    alpha_p = _step_to_boundary(s, ds)
    alpha_d = _step_to_boundary(lam, dlam)
    alpha = jnp.minimum(alpha_p, alpha_d) * _STEP_SAFETY

    x_new = x + alpha * dx
    s_new = s + alpha * ds
    lam_new = lam + alpha * dlam
    return x_new, s_new, lam_new


@partial(jax.jit, static_argnames=("max_iter",))
def solve_qp_pdipm(
    P: Array,
    q: Array,
    G: Array,
    h: Array,
    warm_start: Optional[PdipmState] = None,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> Tuple[Array, Array, PdipmState]:
    """Mehrotra predictor-corrector PDIPM for min 0.5 x^T P x + q^T x s.t. G x <= h.

    The outer loop always runs exactly ``max_iter`` iterations (no early
    termination) so the JIT-compiled function has fixed shape. Quadratic
    convergence near the optimum means residuals saturate to machine
    precision in the last few iters at near-zero cost.

    Returns:
        (x, status, state) where status is:
            1 if final combined residual < tol (solved)
            2 otherwise (max_iter exhausted without meeting tol)
        state is a PdipmState usable as warm_start on subsequent calls.
        Note: state.iter_num is always max_iter (the iteration budget
        consumed), not the iteration at which convergence was first achieved.
    """
    n = P.shape[0]
    m = G.shape[0]

    # --- Initialization ---
    if warm_start is None:
        x0 = jnp.zeros(n)
        s0 = jnp.maximum(h - G @ x0, 1.0)
        lam0 = jnp.ones(m)
    else:
        x0 = warm_start.x
        s0 = jnp.maximum(warm_start.s, 1e-2)
        lam0 = jnp.maximum(warm_start.dual, 1e-2)

    # --- Outer loop (fixed iterations) ---
    def body(_, carry):
        x_, s_, lam_ = carry
        return _pdipm_iteration(P, q, G, h, x_, s_, lam_)

    x_final, s_final, lam_final = lax.fori_loop(0, max_iter, body, (x0, s0, lam0))

    # --- Status: solved if final residual norm below tol ---
    r_d = P @ x_final + G.T @ lam_final + q
    r_p = G @ x_final + s_final - h
    mu = jnp.sum(s_final * lam_final) / m
    res_norm = jnp.max(jnp.abs(r_d)) + jnp.max(jnp.abs(r_p)) + mu
    status = jnp.where(res_norm < tol, jnp.int32(1), jnp.int32(2))

    state = PdipmState(x=x_final, s=s_final, dual=lam_final, iter_num=max_iter)
    return x_final, status, state
