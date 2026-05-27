"""
risk_aware_margins.py
=====================

Closed-form risk-aware safety margins for discrete-time stochastic CBFs, from
Hoxha et al., "Bayesian Risk-Aware CBFs for Discrete-Time Stochastic Systems
with Learned Dynamics" (ACC 2026).

Each margin Delta is the deterministic buffer that must be reserved for the
stochastic term so that the probability of leaving the safe set over horizon T
is at most rho_d. They are plugged into the discrete-time RA-CBF barrier
condition as ``beta = Delta / T`` (see ``discrete_time_risk_aware_barrier``).

Functions
---------
- ct_margin(eta, time_horizon, rho_d):        Eq. 8   -> RA-CBF-CT
- dt_robust_margin(eta, time_horizon, rho_d): Eq. 13  -> RA-CBF-DT
- dt_tight_margin(eta, time_horizon, rho_d):  Prop IV.1 (recovers CT under the pathwise assumption)
"""

import math


def _validate(eta: float, time_horizon: float, rho_d: float) -> None:
    if not eta > 0:
        raise ValueError(f"eta must be > 0, got {eta}")
    if not 0 < time_horizon < math.inf:
        raise ValueError(f"time_horizon must be in (0, inf), got {time_horizon}")
    if not 0 < rho_d < 1:
        raise ValueError(f"rho_d must be in (0, 1), got {rho_d}")


def ct_margin(eta: float, time_horizon: float, rho_d: float) -> float:
    """Continuous-time risk margin (Eq. 8): sqrt(2)*eta*sqrt(T)*erfc^{-1}(rho_d).

    Uses erfc^{-1}(rho_d) = erfinv(1 - rho_d).
    """
    _validate(eta, time_horizon, rho_d)
    erfc_inv = _erfinv(1.0 - rho_d)
    return math.sqrt(2.0) * eta * math.sqrt(time_horizon) * erfc_inv


def dt_robust_margin(eta: float, time_horizon: float, rho_d: float) -> float:
    """Discrete-time robust margin (Eq. 13): sqrt(2*eta^2*T*ln(1/rho_d))."""
    _validate(eta, time_horizon, rho_d)
    return math.sqrt(2.0 * eta**2 * time_horizon * math.log(1.0 / rho_d))


def dt_tight_margin(eta: float, time_horizon: float, rho_d: float) -> float:
    """Tighter discrete-time margin (Prop IV.1) — equals the CT margin under the
    pathwise noise-sensitivity assumption.

    In this phase it is *intentionally identical* to ``ct_margin`` (not a
    placeholder); kept as a distinct name so callers can express the Prop IV.1
    semantics explicitly. Validates its own inputs for symmetry with the
    sibling margins.
    """
    _validate(eta, time_horizon, rho_d)
    return ct_margin(eta, time_horizon, rho_d)


def _erfinv(y: float) -> float:
    """Inverse error function for scalar y in (-1, 1), via Newton refinement of a
    rational approximation. Pure-Python (these margins are scalars computed once
    per controller build; no JAX tracing needed).

    Targets the moderate-tail regime used by these risk margins (accurate to
    ~1e-4 there); it is NOT a drop-in high-precision ``erfinv`` and loses
    precision as ``|y| -> 1``. The validated call path (ct_margin with
    rho_d in (0, 1)) never reaches the degenerate tail."""
    if not -1.0 < y < 1.0:
        raise ValueError(f"erfinv domain is (-1, 1), got {y}")
    a = 0.147
    ln = math.log(1.0 - y * y)
    term = 2.0 / (math.pi * a) + ln / 2.0
    x = math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), y)
    for _ in range(2):
        x -= (math.erf(x) - y) / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))
    return x
