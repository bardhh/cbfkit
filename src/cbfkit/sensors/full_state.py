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

    Returns:
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
    """Senses the state subject to additive, unbiased (zero-mean), Gaussian
    noise.

    Args:
        t (float): time (sec)
        x (Array): state vector (ground truth)
        sigma (Array): measurement model covariance matrix

    Returns:
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

    def sample_mean(num_samples, rng_key):
        # Generate samples: (num_samples, dim)
        # random.normal creates (num_samples, dim) directly
        samples = random.normal(rng_key, shape=(num_samples, dim))

        # Transform: (chol @ samples.T).T -> samples @ chol.T
        transformed = jnp.dot(samples, chol.T)

        # Mean across samples
        vec = jnp.mean(transformed, axis=0)
        return vec

    # Logic: if t == 0, take 10 samples, else 1 sample
    # We rely on lax.cond for JIT compatibility
    n_initial_meas = 10

    sampled_random_vector = lax.cond(
        t == 0.0, lambda k: sample_mean(n_initial_meas, k), lambda k: sample_mean(1, k), key
    )

    sampled_random_vector = sampled_random_vector.reshape(x.shape)

    return x + sampled_random_vector
