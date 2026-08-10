"""Tests for the opt-in ``report_failures`` flag on the CBF-CLF-QP generator.

The failure reporter is built from ``jax.debug`` host callbacks. XLA cannot elide those,
so leaving them in the graph costs time on every step -- including successful ones. These
tests pin the contract: the default graph is callback-free, the flag brings the reporter
back, and ``sub_data["solver_status"]`` is recorded either way (it is consumed by
``simulation/status.py`` and ``simulation/safety_verification.py``).
"""

import jax
import jax.numpy as jnp
import pytest

from cbfkit.controllers.cbf_clf import (
    risk_aware_cbf_clf_qp_controller,
    risk_aware_path_integral_cbf_clf_qp_controller,
    robust_cbf_clf_qp_controller,
    stochastic_cbf_clf_qp_controller,
    vanilla_cbf_clf_qp_controller,
)
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_vanilla_clf_constraints,
    generate_compute_zeroing_cbf_constraints,
)
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.utils.user_types import CertificateCollection, ControllerData

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAFE_STATE = jnp.array([3.0, 3.0])
U_NOM = jnp.array([-0.5, -0.5])
KEY = jax.random.PRNGKey(0)


def _single_integrator(x):
    """2D single integrator: xdot = u."""
    return jnp.zeros(2), jnp.eye(2)


def _unit_circle_barrier():
    """h(x) = ||x||^2 - 1 >= 0 (obstacle of radius 1 at the origin)."""
    return CertificateCollection(
        [lambda t, x: jnp.sum(x**2) - 1.0],
        [lambda t, x: 2.0 * x],
        [lambda t, x: 2.0 * jnp.eye(2)],
        [lambda t, x: 0.0],
        [lambda val: val],
    )


def _build(**extra):
    """A vanilla CBF-CLF-QP controller on the 2D single integrator."""
    return vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0]),
        dynamics_func=_single_integrator,
        barriers=_unit_circle_barrier(),
        **extra,
    )


def _jaxpr_text(controller, x=SAFE_STATE):
    return str(jax.make_jaxpr(controller)(0.0, x, U_NOM, KEY, ControllerData()))


def _build_infeasible(**extra):
    """A 1D controller whose QP is primal-infeasible, so the reporter fires."""

    def dynamics(x):
        return jnp.zeros(1), jnp.ones((1, 1))

    # h(x) = -1 everywhere with a zero gradient: the CBF constraint can never be met.
    barriers = CertificateCollection(
        [lambda t, x: -1.0],
        [lambda t, x: jnp.array([0.0])],
        [lambda t, x: jnp.array([[0.0]])],
        [lambda t, x: 0.0],
        [lambda val: val],
    )
    return vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0]),
        dynamics_func=dynamics,
        barriers=barriers,
        slack_bound_cbf=1e-3,
        relaxable_cbf=False,
        **extra,
    )


def _run_infeasible(controller):
    return controller(0.0, jnp.zeros(1), jnp.zeros(1), KEY, ControllerData())


# ---------------------------------------------------------------------------
# Jaxpr shape of the compiled graph
# ---------------------------------------------------------------------------


def test_default_jaxpr_has_no_debug_callbacks():
    """AC1: the default controller graph contains no host callbacks."""
    text = _jaxpr_text(_build())
    assert "debug_callback" not in text
    assert "debug_print" not in text


def test_report_failures_puts_callbacks_back_in_jaxpr():
    """AC2: the flag restores the reporter in the traced graph."""
    text = _jaxpr_text(_build(report_failures=True))
    assert "debug_callback" in text


def test_reporter_is_the_only_difference():
    """The flag adds callbacks and nothing else removes them."""
    off = _jaxpr_text(_build())
    on = _jaxpr_text(_build(report_failures=True))
    assert off.count("debug_callback") == 0
    assert on.count("debug_callback") > 0
    # The flag only ever grows the graph; it must not alter the compute otherwise.
    assert len(on) > len(off)


# ---------------------------------------------------------------------------
# Printing behavior
# ---------------------------------------------------------------------------


def test_report_failures_prints_on_failure(capfd):
    """AC2: with the flag on, a failed solve still emits the diagnostic."""
    controller = _build_infeasible(report_failures=True)
    _u, data = _run_infeasible(controller)
    jax.effects_barrier()

    out = capfd.readouterr().out
    assert "CBF-CLF-QP Failed" in out
    assert bool(data.error) is True


def test_default_is_silent_on_failure(capfd):
    """The default path fails just as loudly in the data, and silently on stdout."""
    controller = _build_infeasible()
    _u, data = _run_infeasible(controller)
    jax.effects_barrier()

    out = capfd.readouterr().out
    assert "CBF-CLF-QP Failed" not in out
    assert bool(data.error) is True


# ---------------------------------------------------------------------------
# solver_status telemetry survives the gate (AC3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("report_failures", [False, True])
def test_solver_status_recorded_on_success(report_failures):
    controller = _build(report_failures=report_failures)
    _u, data = controller(0.0, SAFE_STATE, U_NOM, KEY, ControllerData())

    assert "solver_status" in data.sub_data
    assert int(data.sub_data["solver_status"]) == 1
    assert "solver_iter" in data.sub_data
    assert bool(data.error) is False


@pytest.mark.parametrize("report_failures", [False, True])
def test_solver_status_recorded_on_failure(report_failures):
    """status.py and safety_verification.py read this key to count failures."""
    controller = _build_infeasible(report_failures=report_failures)
    _u, data = _run_infeasible(controller)

    assert "solver_status" in data.sub_data
    status = int(data.sub_data["solver_status"])
    assert status != 1, "infeasible QP must not report success"
    # error_data mirrors the status for the status.py fallback path.
    assert int(data.error_data) == status
    assert bool(data.error) is True


def test_gating_does_not_change_the_control_output():
    """The flag is diagnostics-only: same control, same status, either way."""
    u_off, data_off = _build()(0.0, SAFE_STATE, U_NOM, KEY, ControllerData())
    u_on, data_on = _build(report_failures=True)(0.0, SAFE_STATE, U_NOM, KEY, ControllerData())

    assert jnp.allclose(u_off, u_on, atol=1e-10)
    assert int(data_off.sub_data["solver_status"]) == int(data_on.sub_data["solver_status"])


# ---------------------------------------------------------------------------
# The flag is reachable from every variant entry point
# ---------------------------------------------------------------------------


def _ra_params():
    return RiskAwareParams(
        t_max=1.0,
        p_bound=0.05,
        gamma=jnp.array([0.5]),
        eta=2.0,
        epsilon=0.1,
        lambda_h=1.0,
        lambda_generator=1.0,
        sigma=lambda x: jnp.zeros((x.shape[0], 1)),
        varsigma=lambda x: jnp.zeros((x.shape[0], x.shape[0])),
        integrator_states=jnp.zeros((1,)),
    )


def _origin_lyapunov():
    """V(x) = ||x||^2 with Vdot <= -V."""
    return CertificateCollection(
        [lambda t, x: jnp.sum(x**2)],
        [lambda t, x: 2.0 * x],
        [lambda t, x: 2.0 * jnp.eye(2)],
        [lambda t, x: 0.0],
        [lambda val: -val],
    )


# Each variant gets the certificate set it actually supports. risk_aware is CLF-only:
# generate_compute_ra_cbf_constraints raises NotImplementedError when given barriers,
# so the barrier path there is deliberately unreachable.
VARIANTS = {
    "vanilla": (vanilla_cbf_clf_qp_controller, {"barriers": _unit_circle_barrier()}),
    "robust": (
        robust_cbf_clf_qp_controller,
        {
            "barriers": _unit_circle_barrier(),
            "disturbance_norm_bound": 0.1,
            "disturbance_norm": 2,
        },
    ),
    "stochastic": (
        stochastic_cbf_clf_qp_controller,
        {"barriers": _unit_circle_barrier(), "sigma": lambda x: jnp.zeros((2, 1))},
    ),
    "risk_aware": (
        risk_aware_cbf_clf_qp_controller,
        {"lyapunovs": _origin_lyapunov(), "ra_clf_params": _ra_params()},
    ),
    "risk_aware_path_integral": (
        risk_aware_path_integral_cbf_clf_qp_controller,
        {"barriers": _unit_circle_barrier(), "ra_cbf_params": _ra_params()},
    ),
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_flag_reaches_every_variant(name):
    """AC1/AC2 across vanilla, robust, stochastic, risk_aware, risk_aware_path_integral."""
    factory, extra = VARIANTS[name]

    def make(**flag):
        return factory(
            control_limits=jnp.array([1.0, 1.0]),
            dynamics_func=_single_integrator,
            **extra,
            **flag,
        )

    assert "debug_callback" not in _jaxpr_text(make()), f"{name} leaks callbacks by default"
    assert "debug_callback" in _jaxpr_text(
        make(report_failures=True)
    ), f"{name} cannot reach report_failures"


def test_flag_is_not_forwarded_into_constraint_kwargs():
    """``report_failures`` is consumed by the generator, not leaked into **kwargs."""
    seen = {}

    def spy_generator(control_limits, dyn_func, barriers, lyapunovs, **kwargs):
        seen.update(kwargs)
        return lambda t, x: (jnp.zeros((0, 2)), jnp.zeros((0,)), {"complete": True})

    generate = cbf_clf_qp_generator(spy_generator, generate_compute_vanilla_clf_constraints)
    generate(
        control_limits=jnp.array([1.0, 1.0]),
        dynamics_func=_single_integrator,
        barriers=_unit_circle_barrier(),
        report_failures=True,
    )

    assert "report_failures" not in seen


def test_default_remains_off():
    """Regression guard: the reporter must stay opt-in."""
    import inspect

    sig = inspect.signature(
        cbf_clf_qp_generator(
            generate_compute_zeroing_cbf_constraints,
            generate_compute_vanilla_clf_constraints,
        )
    )
    assert sig.parameters["report_failures"].default is False
