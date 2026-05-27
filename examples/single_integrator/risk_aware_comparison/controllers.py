"""Builds the four Fig. 1 controllers.

- nominal:   pass-through (no safety filter).
- s_cbf:     per-step stochastic CBF (supermartingale; conservative) via stochastic_cbf_clf_qp_controller.
- ra_cbf_dt / ra_cbf_ct: accumulating-budget path-integral Risk-Aware CBF (Eq. 13 / Eq. 8 margin),
  using the core ``accumulating_risk_aware_cbf_controller``.

S-CBF and RA-CBF are intentionally different controller families: a per-step pointwise condition
vs a finite-horizon accumulated-drift budget. The accumulating form is required to reproduce the
paper's failure rates (a per-step RA-CBF damps the noise it should let through). The accumulating
controller is JIT-safe (carries its integral in ControllerData), so the harness can run with JIT.
"""
import jax.numpy as jnp
from jax import Array

from cbfkit.certificates import concatenate_certificates, generate_certificate
from cbfkit.certificates.conditions.barrier_conditions import stochastic_barrier
from cbfkit.controllers.cbf_clf import (
    accumulating_risk_aware_cbf_controller,
    stochastic_cbf_clf_qp_controller,
)
from cbfkit.controllers.utils import setup_controller
from cbfkit.optimization.quadratic_program import get_solver

from examples.single_integrator.risk_aware_comparison import config as c
from examples.single_integrator.risk_aware_comparison.barrier import make_barrier, make_cost_barrier
from examples.single_integrator.risk_aware_comparison.dynamics import make_dynamics, make_sigma

CONTROLLER_NAMES = ("nominal", "s_cbf", "ra_cbf_dt", "ra_cbf_ct")


def _passthrough_controller():
    """Safety 'filter' that returns u_nom unchanged (for the Nominal baseline)."""

    def passthrough(t: float, x: Array, u_nom: Array) -> Array:
        return u_nom

    return setup_controller(passthrough)


def _stochastic_cbf_controller(condition):
    """Per-step stochastic CBF on the reframed h >= 0 barrier."""
    return stochastic_cbf_clf_qp_controller(
        control_limits=jnp.array([c.ACTUATION_LIMIT, c.ACTUATION_LIMIT]),
        dynamics_func=make_dynamics(),
        barriers=concatenate_certificates(
            generate_certificate(make_barrier(c.R_C), certificate_conditions=condition)
        ),
        sigma=make_sigma(c.SIGMA),
        relaxable_cbf=True,
        solver=get_solver(
            "fast"
        ),  # PDIPM: robust+fast on slack-relaxed QPs (OSQP thrashes near origin)
    )


def _accumulating_ra_controller(margin: str):
    """Accumulating path-integral RA-CBF on the cost barrier B < 1 (core controller)."""
    gamma = (c.X0[0] ** 2 + c.X0[1] ** 2) / c.R_C**2  # B(x0)
    return accumulating_risk_aware_cbf_controller(
        make_dynamics(),
        make_cost_barrier(c.R_C),
        sigma=make_sigma(c.SIGMA),
        control_limits=jnp.array([c.ACTUATION_LIMIT, c.ACTUATION_LIMIT]),
        rho_d=c.RHO_D,
        eta=c.ETA,
        time_horizon=c.T,
        gamma=gamma,
        dt=c.DT,
        margin=margin,
    )


def build_controller(name: str):
    """Return a ControllerCallable for the given controller name."""
    if name == "nominal":
        return _passthrough_controller()
    if name == "s_cbf":
        return _stochastic_cbf_controller(
            stochastic_barrier.right_hand_side(c.SCBF_ALPHA, c.SCBF_BETA)
        )
    if name == "ra_cbf_dt":
        return _accumulating_ra_controller("dt_robust")
    if name == "ra_cbf_ct":
        return _accumulating_ra_controller("ct")
    raise ValueError(f"unknown controller {name!r}; expected one of {CONTROLLER_NAMES}")
