"""Cross-solver objective-convention parity tests.

Registry convention (see ``solver_registry`` module docstring): every solver
returned by ``get_solver()`` solves

    min_x  x^T H x + f^T x   s.t.  G x <= h

Historically the jaxopt wrapper adapted ``(H, 0.5 f)`` into OSQP's
``min 1/2 x^T Q x + c^T x`` form, while the fast (PDIPM) and cvxopt registry
wrappers passed ``(H, f)`` through unchanged — silently solving
``min 1/2 x^T H x + f^T x`` instead. For the CBF-CLF-QP generator (which
passes ``f = -2 H u_nom``) that meant tracking ``2 u_nom`` instead of
``u_nom`` whenever constraints were inactive. These tests pin the convention
for every registered backend and add a controller-level regression with
geometry chosen so the active set does NOT mask the objective-center error.
"""

import jax.numpy as jnp
import pytest
from jax import random

from cbfkit.optimization.quadratic_program import get_solver

JIT_SOLVERS = ["jaxopt", "fast"]
ALL_SOLVERS = ["jaxopt", "fast", "cvxopt", "casadi"]


def _get_solver_or_skip(name: str):
    if name in ("cvxopt", "casadi"):
        try:
            solver = get_solver(name)
            # Import errors surface on first call for lazily-imported backends
            solver(jnp.eye(1), jnp.zeros(1), jnp.ones((1, 1)), jnp.ones(1))
        except ImportError:
            pytest.skip(f"{name} not installed")
        return solver
    return get_solver(name)


# -- Analytic problems in the registry convention -------------------------


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_interior_optimum(name):
    """min ||x - [1,2]||^2 with loose box: H=I, f=-2*[1,2] => x* = [1,2]."""
    solver = _get_solver_or_skip(name)
    H = jnp.eye(2)
    f = -2.0 * jnp.array([1.0, 2.0])
    G = jnp.vstack([jnp.eye(2), -jnp.eye(2)])
    h = 10.0 * jnp.ones(4)
    sol = solver(H, f, G, h)
    assert int(sol.status) == 1
    assert jnp.allclose(
        sol.primal, jnp.array([1.0, 2.0]), atol=2e-3
    ), f"{name}: expected registry-convention minimizer [1,2], got {sol.primal}"


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_single_active_constraint(name):
    """min ||x - [2,0]||^2 s.t. x0 <= 1 => x* = [1,0] (projection onto halfspace)."""
    solver = _get_solver_or_skip(name)
    H = jnp.eye(2)
    f = -2.0 * jnp.array([2.0, 0.0])
    G = jnp.array([[1.0, 0.0]])
    h = jnp.array([1.0])
    sol = solver(H, f, G, h)
    assert int(sol.status) == 1
    assert jnp.allclose(
        sol.primal, jnp.array([1.0, 0.0]), atol=2e-3
    ), f"{name}: expected [1,0], got {sol.primal}"


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_two_active_constraints_corner(name):
    """min ||x - [2,3]||^2 s.t. x <= [1,1] => x* = [1,1] (corner)."""
    solver = _get_solver_or_skip(name)
    H = jnp.eye(2)
    f = -2.0 * jnp.array([2.0, 3.0])
    G = jnp.eye(2)
    h = jnp.ones(2)
    sol = solver(H, f, G, h)
    assert int(sol.status) == 1
    assert jnp.allclose(
        sol.primal, jnp.array([1.0, 1.0]), atol=2e-3
    ), f"{name}: expected [1,1], got {sol.primal}"


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_audit_probe_matches_u_nom(name):
    """The exact probe from the solver audit: H=I, f=-2*u_nom, loose box.

    u_nom = [1.0, -0.5]; pre-fix the fast and cvxopt backends returned
    [2.0, -1.0] with a "solved" status.
    """
    solver = _get_solver_or_skip(name)
    u_nom = jnp.array([1.0, -0.5])
    H = jnp.eye(2)
    f = -2.0 * H @ u_nom
    G = jnp.vstack([jnp.eye(2), -jnp.eye(2)])
    h = 10.0 * jnp.ones(4)
    sol = solver(H, f, G, h)
    assert int(sol.status) == 1
    assert jnp.allclose(sol.primal, u_nom, atol=1e-4), f"{name}: got {sol.primal}"


# -- Random cross-solver agreement (JIT backends) --------------------------


@pytest.mark.parametrize("seed", range(5))
def test_fast_matches_jaxopt_random(seed):
    """fast and jaxopt agree on the primal for random well-conditioned QPs."""
    key = random.PRNGKey(seed)
    k1, k2, k3, k4 = random.split(key, 4)
    n, m = 4, 8
    H = jnp.diag(jnp.abs(random.normal(k1, (n,))) + 0.5)
    f = random.normal(k2, (n,))
    G = random.normal(k3, (m, n))
    h = jnp.abs(random.normal(k4, (m,))) + 0.5

    sols = {}
    for name in JIT_SOLVERS:
        sol = get_solver(name)(H, f, G, h)
        assert int(sol.status) == 1, f"{name} failed on seed {seed}"
        sols[name] = sol.primal

    assert jnp.allclose(
        sols["fast"], sols["jaxopt"], atol=2e-3
    ), f"seed {seed}: fast={sols['fast']} vs jaxopt={sols['jaxopt']}"


# -- Controller-level regression -------------------------------------------


def _make_cbf_controller(solver_name: str, alpha: float = 1.0):
    from cbfkit.certificates import generate_certificate
    from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import (
        linear_class_k,
    )
    from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
    from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator

    dynamics = two_dimensional_single_integrator()

    def h(x):
        return (x[0] - 2.0) ** 2 + x[1] ** 2 - 0.25

    barriers = generate_certificate(h, linear_class_k(alpha), input_style="state")
    return vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([1.0, 1.0]),
        dynamics_func=dynamics,
        barriers=barriers,
        solver=get_solver(solver_name),
    )


@pytest.mark.parametrize("name", JIT_SOLVERS)
def test_cbf_filter_inactive_returns_u_nom(name):
    """With the CBF inactive, the filter must return u_nom — NOT 2*u_nom.

    At x=[0,0]: grad h = [-4, 0], constraint 4*u0 <= h(x) = 3.75, so
    u_nom = [0.3, -0.2] is strictly feasible and must pass through unchanged.
    The pre-fix fast wrapper returned clip(2*u_nom) here.
    """
    from cbfkit.utils.user_types import ControllerData

    controller = _make_cbf_controller(name)
    u, data = controller(
        0.0, jnp.array([0.0, 0.0]), jnp.array([0.3, -0.2]), random.PRNGKey(0), ControllerData()
    )
    assert not bool(data.error)
    assert jnp.allclose(u, jnp.array([0.3, -0.2]), atol=2e-3), f"{name}: u={u}"


@pytest.mark.parametrize("name", JIT_SOLVERS)
def test_cbf_filter_active_non_masking_geometry(name):
    """CBF active with an off-axis nominal so the projection exposes any
    objective-center error.

    At x=[1.2, 0]: h = 0.39, grad h = [-1.6, 0] => constraint u0 <= 0.24375.
    Projecting u_nom=[0.8, 0.6] gives u* = [0.24375, 0.6]. Projecting the
    doubled center [1.6, 1.2] (pre-fix fast) would give u1 = 1.0 (clipped),
    so the u1 component discriminates the conventions.
    """
    from cbfkit.utils.user_types import ControllerData

    expected = jnp.array([0.39 / 1.6, 0.6])
    controller = _make_cbf_controller(name)
    u, data = controller(
        0.0, jnp.array([1.2, 0.0]), jnp.array([0.8, 0.6]), random.PRNGKey(0), ControllerData()
    )
    assert not bool(data.error)
    assert jnp.allclose(u, expected, atol=5e-3), f"{name}: u={u}, expected {expected}"


def test_cbf_filter_audit_case_fast_vs_jaxopt():
    """Audit case u_nom=[1.0, 0.2] at x=[0,0]: both backends must give u[1]=0.2.

    grad h = [-4, 0] involves only u0, so u1 is untouched by the CBF row and
    reports the objective center directly: 0.2 after the fix, 0.4 before it.
    """
    from cbfkit.utils.user_types import ControllerData

    u_nom = jnp.array([1.0, 0.2])
    controls = {}
    for name in JIT_SOLVERS:
        u, data = _make_cbf_controller(name)(
            0.0, jnp.array([0.0, 0.0]), u_nom, random.PRNGKey(0), ControllerData()
        )
        assert not bool(data.error)
        controls[name] = u

    for name, u in controls.items():
        assert jnp.allclose(u[1], 0.2, atol=2e-3), f"{name}: u={u}, expected u[1]=0.2"
    assert jnp.allclose(
        controls["fast"], controls["jaxopt"], atol=2e-3
    ), f"fast={controls['fast']} vs jaxopt={controls['jaxopt']}"


@pytest.mark.parametrize("name", JIT_SOLVERS)
def test_unconstrained_escape_convention(name):
    """No inequality constraints: x* = -(2H)^-1 f in the registry convention."""
    if name == "jaxopt":
        pytest.skip("jaxopt path requires params_ineq or params_eq; escape is fast-only")
    solver = get_solver(name)
    H = jnp.diag(jnp.array([1.0, 4.0]))
    f = jnp.array([-2.0, -8.0])
    sol = solver(H, f)
    assert jnp.allclose(sol.primal, jnp.array([1.0, 1.0]), atol=1e-6), sol.primal
