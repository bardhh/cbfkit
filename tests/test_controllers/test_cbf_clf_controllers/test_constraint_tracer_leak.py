"""Regression tests: constraint generators must not leak tracers into their closures.

Each ``compute_*_constraints`` function is a ``@jit``-wrapped closure over the
zero-initialized constraint template arrays built by ``unpack_for_cbf`` /
``unpack_for_clf``. Those enclosing names used to be rebound through ``nonlocal``
inside the traced body, which stored the first trace's tracers in the closure
cell. Any genuine retrace -- a new input dtype, a new state dimension, or eager
execution under ``jax.disable_jit`` -- then read a dead tracer and raised
``UnexpectedTracerError``. The pattern survived in practice only because
vmap-over-states reuses identical avals and keeps hitting the jit cache.
"""

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.controllers.cbf_clf.generate_constraints import (
    _constraint_core,
    risk_aware_cbfs,
    risk_aware_clfs,
    risk_aware_path_integral_cbfs,
    risk_aware_path_integral_clfs,
)
from cbfkit.controllers.cbf_clf.generate_constraints.risk_aware_cbfs import (
    generate_compute_estimate_feedback_ra_cbf_constraints,
)
from cbfkit.controllers.cbf_clf.generate_constraints.risk_aware_clfs import (
    generate_compute_estimate_feedback_ra_clf_constraints,
    generate_compute_ra_clf_constraints,
)
from cbfkit.controllers.cbf_clf.generate_constraints.risk_aware_path_integral_cbfs import (
    generate_compute_ra_pi_cbf_constraints,
)
from cbfkit.controllers.cbf_clf.generate_constraints.risk_aware_path_integral_clfs import (
    generate_compute_ra_pi_clf_constraints,
)
from cbfkit.controllers.cbf_clf.generate_constraints.robust_cbfs import (
    generate_compute_robust_cbf_constraints,
)
from cbfkit.controllers.cbf_clf.generate_constraints.robust_clfs import (
    generate_compute_robust_clf_constraints,
)
from cbfkit.controllers.cbf_clf.generate_constraints.stochastic_cbfs import (
    generate_compute_stochastic_cbf_constraints,
)
from cbfkit.controllers.cbf_clf.generate_constraints.stochastic_clfs import (
    generate_compute_stochastic_clf_constraints,
)
from cbfkit.controllers.cbf_clf.generate_constraints.vanilla_clfs import (
    generate_compute_vanilla_clf_constraints,
)
from cbfkit.controllers.cbf_clf.generate_constraints.zeroing_cbfs import (
    generate_compute_zeroing_cbf_constraints,
)
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams

N_CON = 2


# The certificates and dynamics below are deliberately written to accept any state
# dimension, so one generator can be called at n=2 and n=3 and be forced to retrace.
def dynamics(x):
    n = x.shape[0]
    return jnp.zeros((n,)), jnp.eye(n)[:, :N_CON]


def h(t, x):
    del t
    return x[0] - 0.1


def dhdx(t, x):
    del t
    return jnp.eye(x.shape[0])[0]


def d2hdx2(t, x):
    del t
    return jnp.zeros((x.shape[0], x.shape[0]))


def dhdt(t, x):
    del t, x
    return 0.0


def alpha(val):
    return 1.0 * val


def sigma(x):
    return 0.25 * jnp.eye(x.shape[0])


CERTS = ([h], [dhdx], [d2hdx2], [dhdt], [alpha])
LIMITS = jnp.array([10.0, 10.0])
LIMITS_SLACK = jnp.array([10.0, 10.0, 10.0])


def _ra_params():
    # ``gamma`` is unused by the plain risk-aware generators but is asserted
    # non-None by the path-integral CLF, so one factory serves every variant.
    return RiskAwareParams(
        t_max=10.0,
        p_bound=0.95,
        eta=2.0,
        epsilon=0.1,
        lambda_h=1.0,
        lambda_generator=1.0,
        gamma=jnp.zeros(1),
        sigma=sigma,
        varsigma=lambda x: 0.1 * jnp.eye(x.shape[0]),
    )


GENERATORS = {
    "zeroing_cbf": lambda: generate_compute_zeroing_cbf_constraints(LIMITS, dynamics, CERTS, CERTS),
    "zeroing_cbf_tunable": lambda: generate_compute_zeroing_cbf_constraints(
        LIMITS_SLACK, dynamics, CERTS, CERTS, tunable_class_k=True
    ),
    "zeroing_cbf_relaxable": lambda: generate_compute_zeroing_cbf_constraints(
        LIMITS_SLACK, dynamics, CERTS, CERTS, relaxable_cbf=True
    ),
    "robust_cbf": lambda: generate_compute_robust_cbf_constraints(
        LIMITS, dynamics, CERTS, CERTS, disturbance_norm_bound=0.3, disturbance_norm=2
    ),
    "stochastic_cbf": lambda: generate_compute_stochastic_cbf_constraints(
        LIMITS, dynamics, CERTS, CERTS, sigma=sigma
    ),
    "vanilla_clf": lambda: generate_compute_vanilla_clf_constraints(LIMITS, dynamics, CERTS, CERTS),
    "vanilla_clf_relaxable": lambda: generate_compute_vanilla_clf_constraints(
        LIMITS_SLACK, dynamics, CERTS, CERTS, relaxable_clf=True
    ),
    "robust_clf": lambda: generate_compute_robust_clf_constraints(
        LIMITS, dynamics, CERTS, CERTS, disturbance_norm_bound=0.3, disturbance_norm=2
    ),
    "stochastic_clf": lambda: generate_compute_stochastic_clf_constraints(
        LIMITS, dynamics, CERTS, CERTS, sigma=sigma
    ),
    "estimate_feedback_ra_cbf": lambda: generate_compute_estimate_feedback_ra_cbf_constraints(
        LIMITS, dynamics, CERTS, CERTS, ra_cbf_params=_ra_params()
    ),
    "ra_clf": lambda: generate_compute_ra_clf_constraints(
        LIMITS, dynamics, CERTS, CERTS, ra_clf_params=_ra_params()
    ),
    "ra_clf_relaxable": lambda: generate_compute_ra_clf_constraints(
        LIMITS_SLACK, dynamics, CERTS, CERTS, relaxable_clf=True, ra_clf_params=_ra_params()
    ),
    "ra_clf_default_params": lambda: generate_compute_ra_clf_constraints(
        LIMITS, dynamics, CERTS, CERTS
    ),
    "estimate_feedback_ra_clf": lambda: generate_compute_estimate_feedback_ra_clf_constraints(
        LIMITS, dynamics, CERTS, CERTS, ra_clf_params=_ra_params()
    ),
    "estimate_feedback_ra_clf_relaxable": (
        lambda: generate_compute_estimate_feedback_ra_clf_constraints(
            LIMITS_SLACK, dynamics, CERTS, CERTS, relaxable_clf=True, ra_clf_params=_ra_params()
        )
    ),
    "ra_pi_cbf": lambda: generate_compute_ra_pi_cbf_constraints(
        LIMITS, dynamics, CERTS, CERTS, ra_cbf_params=_ra_params()
    ),
    "ra_pi_cbf_tunable": lambda: generate_compute_ra_pi_cbf_constraints(
        LIMITS_SLACK, dynamics, CERTS, CERTS, tunable_class_k=True, ra_cbf_params=_ra_params()
    ),
    "ra_pi_cbf_default_params": lambda: generate_compute_ra_pi_cbf_constraints(
        LIMITS, dynamics, CERTS, CERTS
    ),
    "ra_pi_clf": lambda: generate_compute_ra_pi_clf_constraints(
        LIMITS, dynamics, CERTS, CERTS, ra_clf_params=_ra_params()
    ),
    "ra_pi_clf_relaxable": lambda: generate_compute_ra_pi_clf_constraints(
        LIMITS_SLACK, dynamics, CERTS, CERTS, relaxable_clf=True, ra_clf_params=_ra_params()
    ),
    "ra_pi_clf_default_params": lambda: generate_compute_ra_pi_clf_constraints(
        LIMITS, dynamics, CERTS, CERTS
    ),
}

# The path-integral generators additionally used to mutate ``ra_params`` from
# inside the traced body, storing the trace's tracer on the shared params object.
PATH_INTEGRAL = [
    "ra_pi_cbf",
    "ra_pi_cbf_tunable",
    "ra_pi_cbf_default_params",
    "ra_pi_clf",
    "ra_pi_clf_relaxable",
    "ra_pi_clf_default_params",
]

X2 = jnp.array([0.35, -0.75])
X3 = jnp.array([0.35, -0.75, 0.2])


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_retrace_after_first_call_does_not_leak_tracer(name):
    """A second trace with a different dtype must not hit a leaked tracer."""
    fn = GENERATORS[name]()

    a_ref, b_ref, _ = fn(0.0, X2)

    # Different dtype => cache miss => genuine retrace. Before the fix this raised
    # UnexpectedTracerError while reading the closure's leaked constraint matrix.
    a32, b32, _ = fn(0.0, X2.astype(jnp.float32))

    assert np.all(np.isfinite(np.asarray(a32)))
    np.testing.assert_allclose(np.asarray(a32), np.asarray(a_ref), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(b32), np.asarray(b_ref), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_eager_execution_after_trace_matches_traced_result(name):
    """``jax.disable_jit`` runs the body eagerly and must reproduce the traced result."""
    fn = GENERATORS[name]()

    a_ref, b_ref, _ = fn(0.0, X2)
    with jax.disable_jit():
        a_eager, b_eager, _ = fn(0.0, X2)

    np.testing.assert_array_equal(np.asarray(a_eager), np.asarray(a_ref))
    np.testing.assert_array_equal(np.asarray(b_eager), np.asarray(b_ref))


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_new_state_dimension_forces_clean_retrace(name):
    """Calling the same generator at a new state dimension must retrace cleanly."""
    fn = GENERATORS[name]()

    a2, b2, _ = fn(0.0, X2)
    a3, b3, _ = fn(0.0, X3)

    # Constraint shapes are set by the certificate/control counts, not the state size.
    assert a3.shape == a2.shape
    assert b3.shape == b2.shape
    assert np.all(np.isfinite(np.asarray(a3)))
    assert np.all(np.isfinite(np.asarray(b3)))

    # Re-running at the original dimension still reproduces the original values,
    # i.e. the intervening trace did not corrupt the generator's template arrays.
    a2_again, b2_again, _ = fn(0.0, X2)
    np.testing.assert_array_equal(np.asarray(a2_again), np.asarray(a2))
    np.testing.assert_array_equal(np.asarray(b2_again), np.asarray(b2))


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_nested_trace_matches_standalone_call(name):
    """Calling the generator inside an outer jit must not leak the outer trace's tracers."""
    fn = GENERATORS[name]()

    a_ref, b_ref, _ = fn(0.0, X2)

    @jax.jit
    def outer(t, x):
        a, b, _ = fn(t, x)
        return a, b

    a_nested, b_nested = outer(0.0, X2)
    np.testing.assert_array_equal(np.asarray(a_nested), np.asarray(a_ref))
    np.testing.assert_array_equal(np.asarray(b_nested), np.asarray(b_ref))

    # And the standalone call still works afterwards.
    a_after, b_after, _ = fn(0.0, X2)
    np.testing.assert_array_equal(np.asarray(a_after), np.asarray(a_ref))
    np.testing.assert_array_equal(np.asarray(b_after), np.asarray(b_ref))


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_vmap_over_states_still_matches_per_state_calls(name):
    """The vmap path that masked this bug must keep producing the per-state results."""
    fn = GENERATORS[name]()

    states = jnp.stack([X2, X2 + 0.1, X2 - 0.2])
    a_batched, b_batched = jax.vmap(lambda x: fn(0.0, x)[:2])(states)

    for i in range(states.shape[0]):
        a_i, b_i, _ = fn(0.0, states[i])
        np.testing.assert_allclose(np.asarray(a_batched[i]), np.asarray(a_i), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(b_batched[i]), np.asarray(b_i), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "module",
    [
        _constraint_core,
        risk_aware_cbfs,
        risk_aware_clfs,
        risk_aware_path_integral_cbfs,
        risk_aware_path_integral_clfs,
    ],
)
def test_modules_declare_no_nonlocal(module):
    """Guard the defect class itself: no ``nonlocal`` rebinding in these modules."""
    tree = ast.parse(inspect.getsource(module))
    offenders = [node.names for node in ast.walk(tree) if isinstance(node, ast.Nonlocal)]
    assert not offenders, (
        f"{module.__name__} declares nonlocal for {offenders}; rebinding closure state "
        "inside a jit-traced function leaks tracers across traces."
    )


@pytest.mark.parametrize("module", [risk_aware_path_integral_cbfs, risk_aware_path_integral_clfs])
def test_path_integral_body_does_not_mutate_shared_params(module):
    """The traced body must not assign to any ``ra_params`` attribute.

    Assigning ``ra_params.integrator_states`` from inside the ``@jit`` body stored
    the current trace's tracer on the shared ``RiskAwareParams`` object, which the
    next trace then read. The reset now runs on a construction-time template held
    in a local, so nothing traced escapes onto the shared object.
    """
    tree = ast.parse(inspect.getsource(module))
    offenders = [
        f"{ast.unparse(target)} (line {node.lineno})"
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "ra_params"
        # Construction-time zeroing at module/function scope is fine; only the
        # assignments nested inside the traced closure leak.
        and node.col_offset > 4
    ]
    assert (
        not offenders
    ), f"{module.__name__} mutates shared params inside the traced body: {offenders}"


@pytest.mark.parametrize("name", PATH_INTEGRAL)
def test_path_integral_accumulator_is_stateless_across_calls(name):
    """Five sequential calls must not drift: this integrator never accumulates.

    ``integrator_states`` is written in exactly two places -- zeroed once when the
    generator is built, and reset to those same zeros under ``t == 0`` inside the
    body. Nothing anywhere increments it, so its contribution is identically zero
    at every timestep (see the module docstrings). Pinning that here means a future
    change that makes the integral genuinely accumulate has to update this test
    deliberately rather than silently altering the constraint each call.
    """
    inputs = [(0.1 * k, X2 + 0.01 * k) for k in range(5)]

    # Advance both t and x so a real accumulator, or any leaked per-call state,
    # would show up as drift.
    forward = GENERATORS[name]()
    forward_out = [forward(t, x) for t, x in inputs]

    # A fresh generator driven through the same inputs in reverse order must map
    # each (t, x) to the same constraint pair. Any state carried between calls --
    # an accumulating integral included -- would make the answer depend on how
    # many calls came before, and this comparison would fail.
    backward = GENERATORS[name]()
    backward_out = {}
    for t, x in reversed(inputs):
        a, b, _ = backward(t, x)
        backward_out[float(t)] = (np.asarray(a), np.asarray(b))

    for (t, _x), (a_fwd, b_fwd, _) in zip(inputs, forward_out):
        a_bwd, b_bwd = backward_out[float(t)]
        np.testing.assert_array_equal(a_bwd, np.asarray(a_fwd))
        np.testing.assert_array_equal(b_bwd, np.asarray(b_fwd))

    # And the t == 0 reset branch agrees with the steady-state branch, because
    # both read the same construction-time zeros.
    a_zero, b_zero, _ = forward(0.0, X2)
    a_first, b_first, _ = forward_out[0]
    np.testing.assert_array_equal(np.asarray(a_zero), np.asarray(a_first))
    np.testing.assert_array_equal(np.asarray(b_zero), np.asarray(b_first))


@pytest.mark.parametrize("name", PATH_INTEGRAL)
def test_path_integral_params_object_holds_no_tracer_after_use(name):
    """``ra_params.integrator_states`` must stay a concrete array, never a tracer."""
    params = _ra_params()
    kwarg = "ra_cbf_params" if "cbf" in name else "ra_clf_params"
    if "default_params" in name:
        pytest.skip("variant builds its own params internally")

    limits = LIMITS_SLACK if ("tunable" in name or "relaxable" in name) else LIMITS
    extra = {}
    if "tunable" in name:
        extra["tunable_class_k"] = True
    if "relaxable" in name:
        extra["relaxable_clf"] = True
    build = (
        generate_compute_ra_pi_cbf_constraints
        if "cbf" in name
        else generate_compute_ra_pi_clf_constraints
    )
    fn = build(limits, dynamics, CERTS, CERTS, **{kwarg: params}, **extra)

    fn(0.0, X2)
    fn(0.5, X2 + 0.1)

    stored = params.integrator_states
    assert not isinstance(stored, jax.core.Tracer), (
        f"{name} left a {type(stored).__name__} on the shared RiskAwareParams object; "
        "the next trace would read a dead tracer."
    )
    np.testing.assert_array_equal(np.asarray(stored), np.zeros_like(np.asarray(stored)))
