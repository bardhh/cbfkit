"""Training utilities for neural Control Barrier Functions.

Provides loss functions and a training loop that produces a
``CertificateCollection`` ready for use with the CBF-CLF-QP controller.

Usage::

    from cbfkit.modeling.neural_cbf import train_neural_cbf

    cert = train_neural_cbf(
        dynamics_func=dynamics,
        safe_samples=safe_pts,     # (N, state_dim) points in safe set
        unsafe_samples=unsafe_pts, # (M, state_dim) points in unsafe set
        state_dim=2,
        alpha=1.0,
    )
    # cert is a CertificateCollection, drop-in for the QP generator
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

try:
    import optax
except ImportError:
    raise ImportError(
        "Optax is required for neural CBF training. " "Install it with: pip install cbfkit[neural]"
    )

from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import (
    linear_class_k,
)
from cbfkit.certificates.packager import generate_certificate
from cbfkit.utils.user_types import CertificateCollection, DynamicsCallable

from .model import CBFNetwork, make_cbf_callable


def cbf_loss(
    params: dict,
    model: CBFNetwork,
    safe_samples: Array,
    unsafe_samples: Array,
    dynamics_func: DynamicsCallable,
    alpha: float = 1.0,
    margin: float = 0.1,
) -> Tuple[Array, dict]:
    """Compute the CBF training loss.

    The loss has four terms:

    - **safe**: ``h(x) > margin`` for x in safe set (ReLU penalty)
    - **unsafe**: ``h(x) < -margin`` for x in unsafe set (ReLU penalty)
    - **descent**: ``nabla h . f(x) + alpha * h(x) >= 0`` for x in safe set
      (CBF invariance condition, using only the drift term f)
    - **norm**: Light L2 regularization on params

    Args:
        params: Flax model parameters.
        model: CBFNetwork instance.
        safe_samples: Points in the safe set, shape ``(N, state_dim)``.
        unsafe_samples: Points in the unsafe set, shape ``(M, state_dim)``.
        dynamics_func: System dynamics ``(x) -> (f, g)``.
        alpha: Class-K function gain for the CBF condition.
        margin: Desired separation from zero-level set.

    Returns:
        Tuple of ``(total_loss, loss_dict)`` where ``loss_dict`` has per-term values.
    """

    def h(x):
        return model.apply(params, x)

    # --- Safe set: h(x) > margin ---
    h_safe = jax.vmap(h)(safe_samples)
    loss_safe = jnp.mean(jax.nn.relu(margin - h_safe))

    # --- Unsafe set: h(x) < -margin ---
    h_unsafe = jax.vmap(h)(unsafe_samples)
    loss_unsafe = jnp.mean(jax.nn.relu(h_unsafe + margin))

    # --- CBF descent condition: dh/dx . f(x) + alpha * h(x) >= 0 ---
    def descent_violation(x):
        hx, gh = jax.value_and_grad(h)(x)
        f, _g = dynamics_func(x)
        lie_f = jnp.dot(gh, f)
        return jax.nn.relu(-(lie_f + alpha * hx))

    loss_descent = jnp.mean(jax.vmap(descent_violation)(safe_samples))

    # --- L2 regularization ---
    leaves = jax.tree_util.tree_leaves(params)
    loss_reg = 1e-4 * sum(jnp.sum(p**2) for p in leaves)

    total = loss_safe + loss_unsafe + loss_descent + loss_reg

    return total, {
        "safe": loss_safe,
        "unsafe": loss_unsafe,
        "descent": loss_descent,
        "reg": loss_reg,
        "total": total,
    }


def train_neural_cbf(
    dynamics_func: DynamicsCallable,
    safe_samples: Array,
    unsafe_samples: Array,
    state_dim: int,
    alpha: float = 1.0,
    hidden_layers: Sequence[int] = (64, 64),
    activation: str = "tanh",
    learning_rate: float = 1e-3,
    num_epochs: int = 500,
    margin: float = 0.1,
    key: Optional[Array] = None,
    verbose: bool = False,
    certificate_conditions: Optional[Callable] = None,
) -> CertificateCollection:
    """Train a neural CBF and return a ready-to-use CertificateCollection.

    Args:
        dynamics_func: System dynamics ``(x) -> (f, g)``.
        safe_samples: Points in the safe set, shape ``(N, state_dim)``.
        unsafe_samples: Points in the unsafe set, shape ``(M, state_dim)``.
        state_dim: Dimension of the state vector.
        alpha: Class-K function gain.
        hidden_layers: MLP hidden layer sizes.
        activation: Activation function name.
        learning_rate: Adam learning rate.
        num_epochs: Number of training epochs.
        margin: Desired separation from zero-level set.
        key: JAX PRNG key for initialization.
        verbose: Print loss every 100 epochs.
        certificate_conditions: Class-K function for the certificate.
            Defaults to ``linear_class_k(alpha)`` (linear zeroing CBF).
            See also ``cubic_class_k`` in
            ``cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers``.

    Returns:
        A ``CertificateCollection`` with auto-diffed jacobian, hessian,
        and partial, ready for use with the CBF-CLF-QP generator.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    safe_samples = jnp.asarray(safe_samples)
    unsafe_samples = jnp.asarray(unsafe_samples)

    # Initialize model and optimizer
    model = CBFNetwork(hidden_dims=tuple(hidden_layers), activation=activation)
    params = model.init(key, jnp.zeros(state_dim))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Training loop — samples passed as explicit args to avoid JIT retracing
    # if called again with different-shaped data.
    @jax.jit
    def train_step(params, opt_state, safe, unsafe):
        (loss, info), grads = jax.value_and_grad(cbf_loss, has_aux=True)(
            params, model, safe, unsafe, dynamics_func, alpha, margin
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, info

    for epoch in range(num_epochs):
        params, opt_state, info = train_step(params, opt_state, safe_samples, unsafe_samples)
        if verbose and (epoch % 100 == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch:4d} | "
                f"total={float(info['total']):.4f} "
                f"safe={float(info['safe']):.4f} "
                f"unsafe={float(info['unsafe']):.4f} "
                f"descent={float(info['descent']):.4f}"
            )

    # Build the frozen callable
    h_func = make_cbf_callable(model, params)

    # Default condition: linear class-K
    if certificate_conditions is None:
        certificate_conditions = linear_class_k(alpha)

    # Use the existing pipeline to auto-diff and package
    cert = generate_certificate(
        certificate=h_func,
        certificate_conditions=certificate_conditions,
        input_style="state",
    )

    return cert
