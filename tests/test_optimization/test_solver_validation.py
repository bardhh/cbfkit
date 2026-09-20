"""Malformed QPs must not silently become different, successful problems."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from cbfkit.optimization.quadratic_program import get_solver


@pytest.fixture(params=["fast", "jaxopt", "cvxopt", "casadi"])
def solver(request):
    return get_solver(request.param)


@pytest.mark.parametrize("given", ["g_mat", "h_vec"])
def test_incomplete_inequality_is_rejected(solver, given):
    value = jnp.ones((1, 2)) if given == "g_mat" else jnp.ones(1)
    with pytest.raises(ValueError, match="supplied together"):
        solver(jnp.eye(2), jnp.zeros(2), **{given: value})


@pytest.mark.parametrize("given", ["a_mat", "b_vec"])
def test_incomplete_equality_is_rejected(solver, given):
    value = jnp.ones((1, 2)) if given == "a_mat" else jnp.ones(1)
    error = NotImplementedError if solver.solver_name == "fast" else ValueError
    with pytest.raises(error):
        solver(jnp.eye(2), jnp.zeros(2), **{given: value})


@pytest.mark.parametrize(
    "h_shape,f_shape,g_shape,b_shape",
    [
        ((2, 3), (2,), (1, 2), (1,)),
        ((2, 2), (3,), (1, 2), (1,)),
        ((2, 2), (2, 1), (1, 2), (1,)),
        ((0, 0), (0,), (1, 0), (1,)),
        ((2, 2), (2,), (1, 3), (1,)),
        ((2, 2), (2,), (2,), (1,)),
        ((2, 2), (2,), (1, 2), (2,)),
        ((2, 2), (2,), (1, 2), (1, 1)),
    ],
)
def test_incompatible_shapes_are_rejected(solver, h_shape, f_shape, g_shape, b_shape):
    with pytest.raises(ValueError):
        solver(jnp.ones(h_shape), jnp.zeros(f_shape), jnp.ones(g_shape), jnp.ones(b_shape))


@pytest.mark.parametrize("name", ["fast", "jaxopt"])
def test_incomplete_inequality_is_rejected_under_jit(name):
    solver = get_solver(name)

    @jax.jit
    def call(h, f, g):
        return solver(h, f, g_mat=g).primal

    with pytest.raises(ValueError, match="supplied together"):
        call(jnp.eye(2), jnp.zeros(2), jnp.ones((1, 2)))


@pytest.mark.parametrize("name", ["jaxopt", "cvxopt", "casadi"])
@pytest.mark.parametrize("a_shape,b_shape", [((1, 3), (1,)), ((1, 2), (2,)), ((2,), (1,))])
def test_equality_dimensions_are_checked(name, a_shape, b_shape):
    with pytest.raises(ValueError):
        get_solver(name)(jnp.eye(2), jnp.zeros(2), a_mat=jnp.ones(a_shape), b_vec=jnp.ones(b_shape))


@pytest.mark.parametrize("name", ["fast", "jaxopt"])
def test_incompatible_shapes_are_rejected_under_jit(name):
    solver = get_solver(name)

    @jax.jit
    def call(h, f):
        return solver(h, f, jnp.ones((1, 2)), jnp.ones(1)).primal

    with pytest.raises(ValueError, match="h_mat"):
        call(jnp.eye(3), jnp.zeros(2))


@pytest.mark.parametrize("name", ["fast", "jaxopt"])
@pytest.mark.parametrize("compiled", [False, True])
def test_nonfinite_constrained_problem_cannot_succeed(name, compiled):
    solver = get_solver(name, max_iter=20)

    def call(f):
        return solver(jnp.eye(1), f, jnp.ones((1, 1)), jnp.ones(1)).status

    call = jax.jit(call) if compiled else call
    assert int(call(jnp.array([jnp.nan]))) != 1


@pytest.mark.parametrize("backend_status,expected", [(1, 0), (2, 2), (3, 3)])
def test_nonfinite_guard_preserves_backend_failure_codes(monkeypatch, backend_status, expected):
    from jaxopt import OSQP

    def run(self, **kwargs):
        return (
            SimpleNamespace(primal=jnp.array([jnp.nan])),
            SimpleNamespace(status=jnp.asarray(backend_status), iter_num=jnp.asarray(1)),
        )

    monkeypatch.setattr(OSQP, "run", run)
    result = get_solver("jaxopt")(jnp.eye(1), jnp.zeros(1), jnp.ones((1, 1)), jnp.ones(1))
    assert int(result.status) == expected


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("empty_constraints", [False, True])
def test_fast_unconstrained_failure_status(compiled, empty_constraints):
    solver = get_solver("fast")

    def call(h, f):
        kwargs = {"g_mat": jnp.empty((0, 1)), "h_vec": jnp.empty(0)} if empty_constraints else {}
        result = solver(h, f, **kwargs)
        return result.primal, result.status

    call = jax.jit(call) if compiled else call
    primal, status = call(jnp.zeros((1, 1)), jnp.array([-1.0]))
    assert not jnp.all(jnp.isfinite(primal))
    assert int(status) != 1
    primal, status = call(jnp.eye(1), jnp.array([-4.0]))
    assert int(status) == 1
    assert jnp.allclose(primal, jnp.array([2.0]))
