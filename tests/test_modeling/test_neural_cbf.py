"""Tests for the neural CBF module."""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from cbfkit.modeling.neural_cbf.model import CBFNetwork, create_neural_cbf, make_cbf_callable
from cbfkit.modeling.neural_cbf.training import cbf_loss, train_neural_cbf
from cbfkit.utils.user_types import CertificateCollection


# -- Simple 2D system: single integrator (xdot = u) -------------------------


def _si_dynamics(x):
    """Single integrator: f(x) = 0, g(x) = I."""
    return jnp.zeros(2), jnp.eye(2)


def _generate_samples(key, center, radius, n_safe=200, n_unsafe=100):
    """Generate safe/unsafe samples around a circular obstacle.

    Safe set: outside the obstacle (||x - center|| > radius)
    Unsafe set: inside the obstacle (||x - center|| < radius)
    """
    k1, k2 = random.split(key)

    # Safe: sample in a ring [radius+0.2, radius+2.0] around obstacle
    angles_safe = random.uniform(k1, (n_safe,), minval=0, maxval=2 * jnp.pi)
    radii_safe = random.uniform(k1, (n_safe,), minval=radius + 0.3, maxval=radius + 2.0)
    safe = center + jnp.stack(
        [radii_safe * jnp.cos(angles_safe), radii_safe * jnp.sin(angles_safe)], axis=1
    )

    # Unsafe: sample inside the obstacle
    angles_unsafe = random.uniform(k2, (n_unsafe,), minval=0, maxval=2 * jnp.pi)
    radii_unsafe = random.uniform(k2, (n_unsafe,), minval=0.0, maxval=radius * 0.8)
    unsafe = center + jnp.stack(
        [radii_unsafe * jnp.cos(angles_unsafe), radii_unsafe * jnp.sin(angles_unsafe)], axis=1
    )

    return safe, unsafe


# -- Model tests --------------------------------------------------------------


class TestCBFNetwork:
    def test_forward_pass(self):
        model = CBFNetwork(hidden_dims=(32, 32))
        params = model.init(random.PRNGKey(0), jnp.zeros(2))
        out = model.apply(params, jnp.array([1.0, 2.0]))
        assert out.shape == ()  # scalar

    def test_differentiable(self):
        model = CBFNetwork(hidden_dims=(16,))
        params = model.init(random.PRNGKey(0), jnp.zeros(2))

        def h(x):
            return model.apply(params, x)

        x = jnp.array([1.0, 0.5])
        grad = jax.grad(h)(x)
        assert grad.shape == (2,)
        assert not jnp.any(jnp.isnan(grad))

    def test_hessian(self):
        model = CBFNetwork(hidden_dims=(16,), activation="tanh")
        params = model.init(random.PRNGKey(0), jnp.zeros(2))

        def h(x):
            return model.apply(params, x)

        x = jnp.array([1.0, 0.5])
        hess = jax.hessian(h)(x)
        assert hess.shape == (2, 2)
        assert not jnp.any(jnp.isnan(hess))


class TestCreateNeuralCBF:
    def test_returns_params_and_callable(self):
        params, h_func = create_neural_cbf(state_dim=2, hidden_layers=[32])
        assert isinstance(params, dict)
        out = h_func(jnp.zeros(2))
        assert out.shape == ()

    def test_custom_activation(self):
        params, h_func = create_neural_cbf(
            state_dim=3, hidden_layers=[16, 16], activation="softplus"
        )
        out = h_func(jnp.ones(3))
        assert out.shape == ()

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            create_neural_cbf(state_dim=2, activation="gelu")


class TestMakeCBFCallable:
    def test_binds_params(self):
        model = CBFNetwork(hidden_dims=(16,))
        params = model.init(random.PRNGKey(0), jnp.zeros(2))
        h_func = make_cbf_callable(model, params)
        out = h_func(jnp.array([1.0, 2.0]))
        assert out.shape == ()


# -- Training tests ------------------------------------------------------------


class TestCBFLoss:
    def test_loss_computes(self):
        key = random.PRNGKey(42)
        center = jnp.array([3.0, 3.0])
        safe, unsafe = _generate_samples(key, center, radius=1.0, n_safe=50, n_unsafe=30)

        model = CBFNetwork(hidden_dims=(32,))
        params = model.init(random.PRNGKey(0), jnp.zeros(2))

        loss, info = cbf_loss(params, model, safe, unsafe, _si_dynamics)
        assert jnp.isfinite(loss)
        assert all(k in info for k in ("safe", "unsafe", "descent", "reg", "total"))

    def test_loss_is_differentiable(self):
        key = random.PRNGKey(42)
        center = jnp.array([3.0, 3.0])
        safe, unsafe = _generate_samples(key, center, radius=1.0, n_safe=50, n_unsafe=30)

        model = CBFNetwork(hidden_dims=(16,))
        params = model.init(random.PRNGKey(0), jnp.zeros(2))

        grads = jax.grad(lambda p: cbf_loss(p, model, safe, unsafe, _si_dynamics)[0])(params)
        leaves = jax.tree_util.tree_leaves(grads)
        assert all(jnp.isfinite(g).all() for g in leaves)


class TestTrainNeuralCBF:
    def test_returns_certificate_collection(self):
        key = random.PRNGKey(42)
        center = jnp.array([3.0, 3.0])
        safe, unsafe = _generate_samples(key, center, radius=1.0)

        cert = train_neural_cbf(
            dynamics_func=_si_dynamics,
            safe_samples=safe,
            unsafe_samples=unsafe,
            state_dim=2,
            alpha=1.0,
            hidden_layers=[32, 32],
            num_epochs=50,
            key=random.PRNGKey(0),
        )

        assert isinstance(cert, CertificateCollection)
        assert len(cert.functions) == 1
        assert len(cert.jacobians) == 1
        assert len(cert.hessians) == 1
        assert len(cert.partials) == 1
        assert len(cert.conditions) == 1

    def test_certificate_callables_work(self):
        key = random.PRNGKey(42)
        center = jnp.array([3.0, 3.0])
        safe, unsafe = _generate_samples(key, center, radius=1.0)

        cert = train_neural_cbf(
            dynamics_func=_si_dynamics,
            safe_samples=safe,
            unsafe_samples=unsafe,
            state_dim=2,
            num_epochs=50,
            key=random.PRNGKey(0),
        )

        t = 0.0
        x = jnp.array([5.0, 5.0])  # safe point, far from obstacle

        h_val = cert.functions[0](t, x)
        j_val = cert.jacobians[0](t, x)
        hess_val = cert.hessians[0](t, x)
        p_val = cert.partials[0](t, x)
        cond_val = cert.conditions[0](h_val)

        assert h_val.shape == ()
        assert j_val.shape == (2,)
        assert hess_val.shape == (2, 2)
        assert p_val.shape == ()
        assert cond_val.shape == ()
        assert jnp.isfinite(h_val)

    def test_integration_with_qp_controller(self):
        """End-to-end: train neural CBF, build controller, run one step."""
        key = random.PRNGKey(42)
        center = jnp.array([3.0, 3.0])
        safe, unsafe = _generate_samples(key, center, radius=1.0)

        cert = train_neural_cbf(
            dynamics_func=_si_dynamics,
            safe_samples=safe,
            unsafe_samples=unsafe,
            state_dim=2,
            num_epochs=100,
            key=random.PRNGKey(0),
        )

        from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
        from cbfkit.utils.user_types import ControllerData

        controller = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([5.0, 5.0]),
            dynamics_func=_si_dynamics,
            barriers=cert,
        )

        x = jnp.array([5.0, 5.0])
        u_nom = jnp.array([1.0, 1.0])
        data = ControllerData()

        u, new_data = controller(0.0, x, u_nom, random.PRNGKey(0), data)

        # Should produce a finite control output (safe point, should succeed)
        assert u.shape == (2,)
        assert jnp.isfinite(u).all(), f"Controller returned {u}"

    def test_custom_conditions(self):
        key = random.PRNGKey(42)
        center = jnp.array([3.0, 3.0])
        safe, unsafe = _generate_samples(key, center, radius=1.0, n_safe=50, n_unsafe=30)

        cubic_k = lambda h: 2.0 * h**3

        cert = train_neural_cbf(
            dynamics_func=_si_dynamics,
            safe_samples=safe,
            unsafe_samples=unsafe,
            state_dim=2,
            num_epochs=20,
            certificate_conditions=cubic_k,
            key=random.PRNGKey(0),
        )

        h_val = cert.functions[0](0.0, jnp.array([5.0, 5.0]))
        cond_val = cert.conditions[0](h_val)
        expected = 2.0 * h_val**3
        assert jnp.allclose(cond_val, expected, atol=1e-6)
