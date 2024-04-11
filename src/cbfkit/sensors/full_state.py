import jax.numpy as jnp
from jax import Array, random
from typing import Optional, Union


def perfect(_t: float, x: Array, **_kwargs) -> Array:
    """Perfect sensor -- returns exactly the state.

    Args:
        t (float): time (sec)
        x (Array): state vector (ground truth)

    Returns:
        x (Array): state vector

    """
    return x


def unbiased_gaussian_noise(
    t: float,
    x: Array,
    sigma: Optional[Union[Array, None]] = None,
    key: Optional[Union[random.PRNGKey, None]] = None,
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
        key = random.PRNGKey(0)

    # Calculate the dimension of the random vector
    dim = sigma.shape[0]

    # Generate a zero-mean Gaussian random vector with unit variance
    # (take n_initial_meas measurements at t = 0)
    n_initial_meas = 10
    max_iter = n_initial_meas if t == 0 else 1
    normal_samples = jnp.zeros((max_iter, dim))
    for ii in range(max_iter):
        key, subkey = random.split(key)
        normal_samples = normal_samples.at[ii, :].set(random.normal(subkey, shape=(dim,)))

    # Apply Cholesky decomposition to convert the unit variance vector to the desired covariance matrix
    if jnp.trace(abs(sigma)) > 0:
        chol = jnp.linalg.cholesky(sigma)
    else:
        chol = jnp.zeros(sigma.shape)

    sampled_random_vector = jnp.mean(jnp.dot(chol, normal_samples.T), axis=1)

    return x + jnp.mean(sampled_random_vector)
