"""Accumulating-budget path-integral Risk-Aware CBF controller (Hoxha et al., ACC 2026).

Correct, JIT/scan-safe implementation of the discrete-time RA-CBF that bounds the
*accumulated generator drift* of a cost-barrier ``B`` (safe set ``{B(x) < 1}``) over a finite
horizon::

    I_L(t) = integral_0^t Gamma_B ds  <=  1 - gamma - Delta_rho

with ``Gamma_B(x, u) = grad B . (f + g u) + 0.5 Tr[sigma(x)^T (d2B/dx2) sigma(x)]``.

Unlike a per-step CBF, the constraint depends on the accumulated drift ``I_L`` (carried in
``ControllerData.sub_data["I_L"]``), not on the current noise-realized ``B`` -- so the
stochastic term accumulates freely and crosses the reserved margin ``Delta_rho`` at
approximately ``rho_d``. This reproduces the discrete-time RA-CBF failure rates that the
per-step / supermartingale stochastic CBFs cannot.

Replaces the non-functional ``risk_aware_path_integral_cbf_clf_qp_controller`` (whose
integral never accumulated). Scope: a single barrier on a control-affine system.

JIT support: this controller is decorated with ``@jit`` internally and is compatible with
``lax.scan`` (all carry state is JAX-traceable). The ``sub_data`` dict ``{"I_L": scalar}``
must match its primed structure across steps. The ``simulator.execute`` path primes the
controller automatically before entering ``lax.scan``, so ``use_jit=True`` works correctly.
"""

from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import Array, grad, hessian, jit

from cbfkit.certificates.conditions.barrier_conditions.risk_aware_margins import (
    ct_margin,
    dt_robust_margin,
)
from cbfkit.utils.user_types import ControllerData

_MARGINS = {"dt_robust": dt_robust_margin, "ct": ct_margin}


def accumulating_risk_aware_cbf_controller(
    dynamics_func: Callable[[Array], Tuple[Array, Array]],
    barrier: Callable[[Array], Array],
    *,
    sigma: Callable[[Array], Array],
    control_limits: Array,
    rho_d: float,
    eta: float,
    time_horizon: float,
    gamma: float,
    dt: float,
    margin: str = "dt_robust",
    barrier_grad: Optional[Callable[[Array], Array]] = None,
    barrier_hess: Optional[Callable[[Array], Array]] = None,
) -> Callable[[float, Array, Array, Array, Optional[ControllerData]], Tuple[Array, ControllerData]]:
    """Build a JIT/scan-safe accumulating path-integral RA-CBF controller.

    Args:
        dynamics_func: ``x -> (f, g)`` control-affine dynamics.
        barrier: scalar cost-barrier ``B(x)`` with safe set ``{B(x) < 1}``. SINGLE scalar
            barrier only -- a vector-valued (multi-barrier) function is unsupported and would
            produce a meaningless constraint without raising.
        sigma: diffusion ``x -> Sigma(x)`` (state_dim x noise_dim).
        control_limits: per-axis actuation bound (clipped symmetrically).
        rho_d: tolerable failure probability in (0, 1).
        eta: noise sensitivity bound.
        time_horizon: finite horizon T (seconds).
        gamma: initial barrier-value bound ``B(x0)``.
        dt: simulation timestep (> 0).
        margin: "dt_robust" (Eq. 13) or "ct" (Eq. 8). The Prop-IV.1 tight margin equals the
            CT margin, so use "ct" for it.
        barrier_grad: optional gradient of ``barrier``; defaults to JAX autodiff.
        barrier_hess: optional Hessian of ``barrier``; defaults to JAX autodiff.

    Returns:
        Canonical ``(t, x, u_nom, key, data) -> (u, ControllerData)``.

    Notes:
        Constraint vs. actuation priority: the projected control is clipped to
        ``control_limits``. If the accumulated budget is already exhausted
        (``I_L > 1 - gamma - Delta_rho``), the required inward control can exceed the
        actuation limits, and the clip takes precedence -- the per-step constraint
        ``Gamma_B <= (budget - I_L)/dt`` is then momentarily violated. This is the standard
        physical-infeasibility case (actuator-limited), not a hard safety guarantee.
    """
    if margin not in _MARGINS:
        raise ValueError(f"margin must be one of {sorted(_MARGINS)}, got {margin!r}")
    if not dt > 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    delta_rho = _MARGINS[margin](eta, time_horizon, rho_d)
    budget = 1.0 - gamma - delta_rho
    grad_b = barrier_grad if barrier_grad is not None else grad(barrier)
    hess_b = barrier_hess if barrier_hess is not None else hessian(barrier)

    @jit
    def controller(
        t: float, x: Array, u_nom: Array, key: Array, data: Optional[ControllerData]
    ) -> Tuple[Array, ControllerData]:
        sub = data.sub_data if (data is not None and data.sub_data is not None) else {}
        i_l = sub["I_L"] if "I_L" in sub else jnp.zeros(())
        i_l = jnp.where(t <= 0.0, jnp.zeros(()), i_l)

        f, g = dynamics_func(x)
        gb = grad_b(x)
        sig = sigma(x)
        trace = 0.5 * jnp.trace(sig.T @ hess_b(x) @ sig)
        drift0 = jnp.dot(gb, f) + trace  # generator drift at u = 0
        a = jnp.matmul(gb, g)  # coefficient on u, shape (m,)
        rhs = (budget - i_l) / dt - drift0
        aa = jnp.dot(a, a)
        viol = jnp.dot(a, u_nom) - rhs
        # Minimal-norm projection of u_nom onto {u : a.u <= rhs}. The 1e-12 guard covers
        # a = 0 (barrier critical point, e.g. grad B = 0): the constraint is then inert and
        # u_nom passes through -- correct, since no control direction affects the drift there.
        u = jnp.where(viol > 0.0, u_nom - (viol / jnp.maximum(aa, 1e-12)) * a, u_nom)
        u = jnp.clip(u, -control_limits, control_limits)

        gamma_b = jnp.dot(a, u) + drift0  # realized generator drift at applied u
        new_i_l = i_l + gamma_b * dt
        return u, ControllerData(u=u, u_nom=u_nom, sub_data={"I_L": new_i_l})

    return controller
