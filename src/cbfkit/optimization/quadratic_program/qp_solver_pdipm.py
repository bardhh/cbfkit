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
