import importlib.util

import jax.numpy as jnp
import pytest


def test_qp_solver_casadi_missing_dependency_has_guidance():
    if importlib.util.find_spec("casadi") is not None:
        pytest.skip("casadi is installed; missing-dependency path not applicable.")

    from cbfkit.optimization.quadratic_program.qp_solver_casadi import solve

    with pytest.raises(ImportError, match=r"cbfkit\[casadi\]"):
        solve(jnp.eye(1), jnp.zeros((1,)))


def test_qp_solver_casadi_smoke_when_available():
    if importlib.util.find_spec("casadi") is None:
        pytest.skip("casadi is not installed in this environment.")

    from cbfkit.optimization.quadratic_program.qp_solver_casadi import solve

    h_mat = jnp.eye(1)
    f_vec = jnp.zeros((1,))
    g_mat = jnp.array([[1.0], [-1.0]])
    h_vec = jnp.array([1.0, 1.0])

    sol, status = solve(h_mat, f_vec, g_mat, h_vec)
    assert sol.shape == (1,)
    assert int(status) in (0, 1)
