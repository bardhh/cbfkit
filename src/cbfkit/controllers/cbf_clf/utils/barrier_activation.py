from functools import partial

import jax
import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=["activation_type", "k"])
def compute_activation_weights(
    state: jax.Array,
    obstacle_positions: jax.Array,
    activation_type: str = "combined",
    k: int = 3,
    radius: float = 2.0,
    smoothness: float = 5.0,
) -> jax.Array:
    """
    Computes activation weights for obstacles based on robot state.
    Returns an array of weights (0.0 to 1.0) for each obstacle.
    """
    # Robot position (assume x, y are first two elements)
    pos = state[:2]

    # Compute distances
    diff = obstacle_positions[:, :2] - pos
    dists = jnp.linalg.norm(diff, axis=1)

    # 1. Radius-based activation (Smooth sigmoid)
    # Active (1.0) when dist < radius
    # Inactive (0.0) when dist > radius
    # Weight = 1 / (1 + exp(s * (dist - radius)))
    # If dist=radius, exp(0)=1 -> w=0.5
    # If dist < radius, exponent is negative, exp -> 0, w -> 1
    radius_weights = 1.0 / (1.0 + jnp.exp(smoothness * (dists - radius)))

    # 2. K-Nearest activation
    # Negative distances for top_k (largest negative = smallest positive)
    _, indices = jax.lax.top_k(-dists, k)
    # Create mask
    k_mask = jnp.zeros_like(dists)
    k_mask = k_mask.at[indices].set(1.0)

    # Combined logic
    # For now, simple multiplication: both must be satisfied?
    # Or does the user want "Top K *within* radius"? Yes, usually.
    # If we just multiply, then if it's in Top K but far away, w -> 0 (due to radius).
    # If it's close but not in Top K, w -> 0 (due to k_mask).

    # What if activation_type is different?
    # Since we used static_argnames, we can branch at trace time.
    if activation_type == "radius":
        return radius_weights
    elif activation_type == "k_nearest":
        return k_mask
    else:
        return radius_weights * k_mask
