"""
discrete_time_risk_aware_barrier.py
===================================

Right-hand side of the discrete-time Risk-Aware CBF (RA-CBF) condition from
Hoxha et al. (ACC 2026), for use with the *stochastic* CBF-CLF-QP controller.

CONVENTION (important): CBFKit uses the ``h >= 0``-safe (zeroing) convention with
a lower-bound constraint. The paper's keep-in barrier ``B(x) = ||x||^2/R_c^2``
(safe when ``B < 1``) is reframed as ``h(x) := 1 - B(x)`` so ``{h > 0}`` is the
safe set. Under this reframing the RA-CBF constraint becomes the standard
zeroing-CBF lower bound

    Lf h + Lg h u + 0.5*Tr[sigma^T (d^2 h/dx^2) sigma] + (h - Delta_rho)/T >= 0

i.e. the class-K term ``alpha(h)`` is replaced by ``(h - Delta_rho)/T``. This is
algebraically ``stochastic_barrier.right_hand_side(alpha=1/T, beta=Delta_rho/T)``,
so the existing ``stochastic_cbf_clf_qp_controller`` supplies the diffusion
(Hessian-trace) term with no new QP machinery.

Delta_rho is the closed-form risk margin: ``dt_robust_margin`` (Eq. 13, RA-CBF-DT)
or ``ct_margin`` (Eq. 8, RA-CBF-CT).

NOTE: This per-step parameterization is a reference building block; it is NOT used by the
ACC 2026 Fig. 1 reproduction. Enforcing the risk-aware condition *every step* over-damps the
closed loop (it re-zeros the budget each step, suppressing the noise that should drive
boundary crossings), collapsing the empirical failure probability toward 0. The figure uses
the finite-horizon accumulated-budget controller
``cbfkit.controllers.cbf_clf.accumulating_risk_aware_cbf_controller`` instead.
"""

from typing import Callable

from jax import Array

from cbfkit.certificates.conditions.barrier_conditions.risk_aware_margins import (
    ct_margin,
    dt_robust_margin,
)

_MARGINS = {"dt_robust": dt_robust_margin, "ct": ct_margin}


def right_hand_side(
    rho_d: float, eta: float, time_horizon: float, *, margin: str = "dt_robust"
) -> Callable[[Array], Array]:
    """Build the discrete-time RA-CBF condition ``h -> (h - Delta_rho)/T``.

    Args:
        rho_d: tolerable risk of leaving the safe set over the horizon, in (0, 1).
        eta: noise sensitivity bound sup||dh/dx . sigma|| over the safe set.
        time_horizon: finite horizon T (seconds).
        margin: "dt_robust" (Eq. 13, more conservative) or "ct" (Eq. 8, tighter).

    Returns:
        Callable mapping the barrier value ``h`` to the class-K replacement term.
    """
    if margin not in _MARGINS:
        raise ValueError(f"margin must be one of {sorted(_MARGINS)}, got {margin!r}")
    delta_rho = _MARGINS[margin](eta, time_horizon, rho_d)
    return lambda h: (h - delta_rho) / time_horizon
