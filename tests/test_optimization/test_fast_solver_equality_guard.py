"""The 'fast' (PDIPM) backend must reject equality constraints, not drop them.

``fast_solver`` accepts ``a_mat``/``b_vec`` for interface compatibility with the
other registry backends, but its underlying PDIPM solves ``Gx <= h`` only. It
used to ignore those arguments and return ``status == 1``, so
``generate_mpc_solver_quadratic_cost_linear_dynamics`` — which encodes the plant
dynamics as equality constraints — got a "successful" trajectory that satisfied
no dynamics at all. A loud failure is the only safe behaviour here.
"""

import jax.numpy as jnp
import pytest

from cbfkit.optimization.quadratic_program import get_solver


def _problem():
    """min ||x - [1,2]||^2 s.t. loose box, plus the equality x0 + x1 = 1."""
    H = jnp.eye(2)
    f = -2.0 * jnp.array([1.0, 2.0])
    G = jnp.vstack([jnp.eye(2), -jnp.eye(2)])
    h = 10.0 * jnp.ones(4)
    a_mat = jnp.array([[1.0, 1.0]])
    b_vec = jnp.array([1.0])
    return H, f, G, h, a_mat, b_vec


class TestFastRejectsEqualityConstraints:
    def test_raises_with_inequalities_present(self):
        H, f, G, h, a_mat, b_vec = _problem()
        with pytest.raises(NotImplementedError, match="equality constraints"):
            get_solver("fast")(H, f, G, h, a_mat, b_vec)

    def test_raises_on_the_no_inequality_path(self):
        """The unconstrained escape branch drops a_mat/b_vec just as silently."""
        H, f, _, _, a_mat, b_vec = _problem()
        with pytest.raises(NotImplementedError, match="equality constraints"):
            get_solver("fast")(H, f, None, None, a_mat, b_vec)

    @pytest.mark.parametrize("which", ["a_mat", "b_vec"])
    def test_raises_when_only_one_half_is_given(self, which):
        """Half a specification is a caller bug; the other backends ignore it."""
        H, f, G, h, a_mat, b_vec = _problem()
        kwargs = {"a_mat": a_mat} if which == "a_mat" else {"b_vec": b_vec}
        with pytest.raises(NotImplementedError):
            get_solver("fast")(H, f, G, h, **kwargs)

    def test_message_names_backend_and_alternatives(self):
        H, f, G, h, a_mat, b_vec = _problem()
        with pytest.raises(NotImplementedError) as exc:
            get_solver("fast")(H, f, G, h, a_mat, b_vec)
        message = str(exc.value)
        assert "fast" in message
        assert "jaxopt" in message or "casadi" in message

    def test_inequality_only_call_still_works(self):
        """The guard must not fire on the ordinary CBF-QP path."""
        H, f, G, h, _, _ = _problem()
        sol = get_solver("fast")(H, f, G, h)
        assert int(sol.status) == 1
        assert jnp.allclose(sol.primal, jnp.array([1.0, 2.0]), atol=1e-4)


class TestEqualityCapableBackendsUnaffected:
    def test_jaxopt_still_enforces_equality(self):
        H, f, G, h, a_mat, b_vec = _problem()
        sol = get_solver("jaxopt")(H, f, G, h, a_mat, b_vec)
        assert int(sol.status) == 1
        assert jnp.allclose(sol.primal @ jnp.array([1.0, 1.0]), 1.0, atol=1e-3)

    def test_cvxopt_still_enforces_equality(self):
        H, f, G, h, a_mat, b_vec = _problem()
        try:
            solver = get_solver("cvxopt")
            sol = solver(H, f, G, h, a_mat, b_vec)
        except ImportError:
            pytest.skip("cvxopt/kvxopt not installed")
        assert int(sol.status) == 1
        assert jnp.allclose(sol.primal @ jnp.array([1.0, 1.0]), 1.0, atol=1e-4)
        # Equality-constrained minimizer of ||x-[1,2]||^2 on x0+x1=1 is [0,1].
        assert jnp.allclose(sol.primal, jnp.array([0.0, 1.0]), atol=1e-4)
