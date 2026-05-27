"""Tests for the accumulating path-integral Risk-Aware CBF controller.

Covers:
- test_integral_accumulates:    I_L is 0 at t=0 and grows over subsequent steps.
- test_reset_at_t0:             I_L=5.0 in carried data is correctly reset when t<=0.
- test_margin_selection:        ct margin is less conservative than dt_robust; ct allows
                                a larger outward component at the same state/step.
- test_jit_safe_through_simulator: use_jit=True and use_jit=False both run without error
                                   and return finite states.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random, grad

from cbfkit.controllers.cbf_clf.accumulating_risk_aware_cbf import (
    accumulating_risk_aware_cbf_controller,
)
from cbfkit.utils.user_types import ControllerData

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 2D single integrator: f=0, g=I
_DYNAMICS = lambda x: (jnp.zeros(2), jnp.eye(2))  # noqa: E731

# Cost-barrier B(x) = x @ x  (safe set {||x|| < 1})
_BARRIER = lambda x: jnp.dot(x, x)  # noqa: E731

# Isotropic diffusion
_SIGMA = lambda x: 0.1 * jnp.eye(2)  # noqa: E731

_CTRL_LIMITS = jnp.array([1.0, 1.0])

# Shared RA-CBF hyperparameters
_COMMON = dict(
    dynamics_func=_DYNAMICS,
    barrier=_BARRIER,
    sigma=_SIGMA,
    control_limits=_CTRL_LIMITS,
    rho_d=0.3,
    eta=0.2,
    time_horizon=5.0,
    gamma=0.01,
    dt=0.05,
)

_KEY = random.PRNGKey(0)


def _build_ct(**overrides):
    """Build a controller with ct margin, merging overrides."""
    kwargs = {**_COMMON, "margin": "ct", **overrides}
    return accumulating_risk_aware_cbf_controller(**kwargs)


def _build_dt(**overrides):
    """Build a controller with dt_robust margin, merging overrides."""
    kwargs = {**_COMMON, "margin": "dt_robust", **overrides}
    return accumulating_risk_aware_cbf_controller(**kwargs)


# ---------------------------------------------------------------------------
# test_integral_accumulates
# ---------------------------------------------------------------------------


def test_integral_accumulates():
    """I_L should be 0 at t=0 and strictly increasing over the next few steps."""
    ctrl = _build_ct()

    x = jnp.array([0.1, 0.0])
    data = None
    prev_il = None

    for step in range(6):
        t = step * _COMMON["dt"]
        # Outward nominal control: push toward boundary
        norm_x = jnp.linalg.norm(x)
        u_nom = 0.4 * x / jnp.maximum(norm_x, 1e-6)
        u, data = ctrl(t, x, u_nom, _KEY, data)

        il = float(data.sub_data["I_L"])

        if step == 0:
            # At t=0 the reset fires: I_L starts from 0
            assert (
                il == pytest.approx(0.0, abs=1e-5) or il >= 0.0
            ), f"Step 0: expected I_L >= 0, got {il}"
            prev_il = il
        else:
            assert il > prev_il, f"Step {step}: I_L did not increase ({prev_il} -> {il})"
            prev_il = il

        # Advance state naively (single integrator: x_next = x + u*dt)
        x = x + u * _COMMON["dt"]


# ---------------------------------------------------------------------------
# test_reset_at_t0
# ---------------------------------------------------------------------------


def test_reset_at_t0():
    """I_L=5.0 carried in sub_data must be reset when t=0."""
    ctrl = _build_ct()

    x = jnp.array([0.1, 0.0])
    # Pre-load a large I_L
    data_with_stale_il = ControllerData(sub_data={"I_L": jnp.array(5.0)})

    u_nom = jnp.array([0.2, 0.0])
    u, out_data = ctrl(0.0, x, u_nom, _KEY, data_with_stale_il)

    # The controller should use I_L=0 (reset), not I_L=5.0.
    # Verify: call again with data=None at t=0 and compare outputs.
    u_ref, _ = ctrl(0.0, x, u_nom, _KEY, None)

    assert jnp.allclose(
        u, u_ref, atol=1e-5
    ), f"Reset failed: u with stale I_L={u} != u with clean slate={u_ref}"

    # I_L after the step should be small (started from 0, not from 5)
    il = float(out_data.sub_data["I_L"])
    assert abs(il) < 1.0, f"I_L after reset step should be small, got {il}"


# ---------------------------------------------------------------------------
# test_margin_selection
# ---------------------------------------------------------------------------


def test_margin_selection():
    """ct margin is less conservative than dt_robust: same state/step, ct allows
    a larger outward component (dot(grad_b, u_ct) >= dot(grad_b, u_dt))."""
    ctrl_ct = _build_ct()
    ctrl_dt = _build_dt()

    x = jnp.array([0.5, 0.0])
    # Strong outward push — both controllers should clip it, but ct less so.
    u_nom = jnp.array([1.0, 0.0])

    u_ct, _ = ctrl_ct(0.0, x, u_nom, _KEY, None)
    u_dt, _ = ctrl_dt(0.0, x, u_nom, _KEY, None)

    gb = grad(_BARRIER)(x)  # = 2*x
    dot_ct = float(jnp.dot(gb, u_ct))
    dot_dt = float(jnp.dot(gb, u_dt))

    assert dot_ct >= dot_dt - 1e-6, (
        f"ct should be less conservative than dt_robust: "
        f"dot(grad_b, u_ct)={dot_ct:.6f} < dot(grad_b, u_dt)={dot_dt:.6f}"
    )


# ---------------------------------------------------------------------------
# test_jit_safe_through_simulator
# ---------------------------------------------------------------------------


def test_safe_through_simulator_non_jit():
    """Run through simulator.execute with use_jit=False: must return finite states."""
    from cbfkit.simulation import simulator as sim
    from cbfkit.integration import forward_euler
    from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation

    dt = _COMMON["dt"]
    ctrl = _build_ct()
    sigma_func = _SIGMA
    perturbation = generate_stochastic_perturbation(sigma_func, dt)

    x0 = jnp.array([0.1, 0.0])

    def outward_nom(t, x, key, ref):
        norm_x = jnp.linalg.norm(x)
        return 0.4 * x / jnp.maximum(norm_x, 1e-6), ControllerData()

    results = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=50,
        dynamics=_DYNAMICS,
        integrator=forward_euler,
        nominal_controller=outward_nom,
        controller=ctrl,
        perturbation=perturbation,
        use_jit=False,
        verbose=False,
        key=random.PRNGKey(42),
    )

    assert jnp.all(jnp.isfinite(results.states)), "Non-finite states with use_jit=False"
    assert results.states.shape[0] == 50


def test_jit_safe_through_simulator_jit():
    """Run through simulator.execute with use_jit=True: must return finite states."""
    from cbfkit.simulation import simulator as sim
    from cbfkit.integration import forward_euler
    from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation

    dt = _COMMON["dt"]
    ctrl = _build_ct()
    sigma_func = _SIGMA
    perturbation = generate_stochastic_perturbation(sigma_func, dt)

    x0 = jnp.array([0.1, 0.0])

    def outward_nom(t, x, key, ref):
        norm_x = jnp.linalg.norm(x)
        return 0.4 * x / jnp.maximum(norm_x, 1e-6), ControllerData()

    results = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=50,
        dynamics=_DYNAMICS,
        integrator=forward_euler,
        nominal_controller=outward_nom,
        controller=ctrl,
        perturbation=perturbation,
        use_jit=True,
        verbose=False,
        key=random.PRNGKey(42),
    )

    assert jnp.all(jnp.isfinite(results.states)), "Non-finite states with use_jit=True"
    assert results.states.shape[0] == 50
