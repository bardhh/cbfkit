from typing import Any, Optional

import jax.numpy as jnp
from jax import Array, lax, random

from cbfkit.utils.user_types import Key, Time


def perfect(
    t: Time,
    x: Array,
    *,
    sigma: Optional[Array] = None,
    key: Optional[Key] = None,
    **_kwargs: Any,
) -> Array:
    """Perfect sensor -- returns exactly the state.

    Args:
        t (float): time (sec)
        x (Array): state vector (ground truth)

    Returns
    -------
        x (Array): state vector
    """
    return x


def unbiased_gaussian_noise(
    t: Time,
    x: Array,
    sigma: Optional[Array] = None,
    key: Optional[Key] = None,
    **kwargs: Any,
) -> Array:
    """Senses the state subject to additive, unbiased (zero-mean), Gaussian noise.

    Note: Previous versions averaged 10 samples at t=0. This behavior has been removed
    for consistency with the noise model definition.

    Args:
        t (float): time (sec)
        x (Array): state vector (ground truth)
        sigma (Array): measurement model covariance matrix

    Returns
    -------
        y (Array): measurement of full state vector
    """
    if sigma is None:
        sigma = 0.1 * jnp.eye((len(x)))

    if key is None:
        key = random.PRNGKey(0)  # type: ignore[assignment]

    # Calculate the dimension of the random vector
    dim = sigma.shape[0]

    # Apply Cholesky decomposition
    # Use lax.cond for JIT compatibility
    chol = lax.cond(
        jnp.trace(jnp.abs(sigma)) > 0,
        lambda s: jnp.linalg.cholesky(s),
        lambda s: jnp.zeros(s.shape),
        sigma,
    )

    # Generate random vector z ~ N(0, I)
    z = random.normal(key, shape=(dim,))

    # Transform to y ~ N(0, Sigma) via y = L @ z
    # chol is lower triangular L such that L @ L.T = Sigma
    sampled_random_vector = jnp.dot(chol, z)

    sampled_random_vector = sampled_random_vector.reshape(x.shape)

    return x + sampled_random_vector
