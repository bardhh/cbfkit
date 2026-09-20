"""Active coverage for equality and infeasibility cases formerly commented out."""

import jax.numpy as jnp
import pytest

from cbfkit.optimization.quadratic_program import get_solver


@pytest.mark.parametrize("name", ["jaxopt", "cvxopt", "casadi"])
@pytest.mark.parametrize("with_inequality", [False, True])
def test_equality_feasible_solution(name, with_inequality):
    # min ||x - [1, 2]||^2, subject to x0 + x1 = 1, has minimizer [0, 1].
    kwargs = {"a_mat": jnp.ones((1, 2)), "b_vec": jnp.ones(1)}
    if with_inequality:
        kwargs.update(g_mat=jnp.eye(2), h_vec=2.0 * jnp.ones(2))
    try:
        result = get_solver(name)(jnp.eye(2), jnp.array([-2.0, -4.0]), **kwargs)
    except ImportError:
        pytest.skip(f"Optional solver {name} is not installed")
    assert int(result.status) == 1
    assert jnp.allclose(result.primal, jnp.array([0.0, 1.0]), atol=1e-3)


@pytest.mark.parametrize("name", ["fast", "jaxopt", "cvxopt", "casadi"])
def test_contradictory_inequalities_do_not_report_success(name):
    # x <= 0 and x >= 1 cannot both hold.
    try:
        result = get_solver(name)(
            jnp.eye(1), jnp.zeros(1), jnp.array([[1.0], [-1.0]]), jnp.array([0.0, -1.0])
        )
    except ImportError:
        pytest.skip(f"Optional solver {name} is not installed")
    except ValueError as exc:
        # CVXOPT can fail its interior-point factorization on this infeasible
        # problem instead of returning a status. Both paths explicitly fail.
        assert name == "cvxopt"
        assert "domain error" in str(exc)
        return
    assert int(result.status) != 1
