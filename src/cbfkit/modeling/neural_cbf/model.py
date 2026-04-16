"""Neural Control Barrier Function model.

Provides a Flax-based MLP that maps state to a scalar barrier value,
and a factory function that produces a frozen callable compatible with
the CBFKit certificate pipeline.

Usage::

    from cbfkit.modeling.neural_cbf import create_neural_cbf

    params, h_func = create_neural_cbf(
        state_dim=2,
        hidden_layers=[64, 64],
        key=jax.random.PRNGKey(0),
    )
    # h_func(x) -> scalar, compatible with generate_certificate
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

try:
    import flax.linen as nn
except ImportError:
    raise ImportError(
        "Flax is required for neural CBFs. " "Install it with: pip install cbfkit[neural]"
    )


class CBFNetwork(nn.Module):
    """MLP that maps state to a scalar barrier value.

    The final layer has no activation, producing an unconstrained scalar.
    Hidden layers use ``tanh`` by default (smooth, so ``jax.grad`` and
    ``jax.hessian`` produce well-behaved derivatives).
    """

    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: Array) -> Array:
        act_fn = _get_activation(self.activation)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = act_fn(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)  # scalar output


def _get_activation(name: str):
    activations = {
        "tanh": nn.tanh,
        "softplus": nn.softplus,
        "sigmoid": nn.sigmoid,
        "swish": nn.swish,
    }
    if name not in activations:
        available = ", ".join(sorted(activations))
        raise ValueError(f"Unknown activation {name!r}. Available: {available}")
    return activations[name]


def create_neural_cbf(
    state_dim: int,
    hidden_layers: Sequence[int] = (64, 64),
    activation: str = "tanh",
    key: Optional[Array] = None,
) -> Tuple[dict, Callable[[Array], Array]]:
    """Create a neural CBF model and return frozen parameters + callable.

    Args:
        state_dim: Dimension of the state vector.
        hidden_layers: Hidden layer sizes for the MLP.
        activation: Activation function name (tanh, softplus, sigmoid, swish).
        key: JAX PRNG key for parameter initialization.

    Returns:
        Tuple of ``(params, h_func)`` where ``h_func(x) -> scalar``
        is a pure JAX function compatible with
        ``generate_certificate(h_func, conditions, input_style="state")``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    model = CBFNetwork(hidden_dims=tuple(hidden_layers), activation=activation)
    params = model.init(key, jnp.zeros(state_dim))
    return params, make_cbf_callable(model, params)


def make_cbf_callable(model: CBFNetwork, params: dict) -> Callable[[Array], Array]:
    """Bind trained parameters to a model, returning ``h(x) -> scalar``.

    Use this after training to create a fresh callable from updated params.
    """

    def h_func(x: Array) -> Array:
        return model.apply(params, x)

    return h_func
