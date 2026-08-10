from typing import Any, Optional

import jax.numpy as jnp
from jax import Array, lax, random

from cbfkit.utils.user_types import Key, SensorCallable, Time


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


def unbiased_gaussian_noise_factory(sigma: Array) -> SensorCallable:
    """Builds an `unbiased_gaussian_noise`-equivalent sensor with the Cholesky factor
    of `sigma` precomputed at construction time.

    `unbiased_gaussian_noise` re-derives the Cholesky factor of `sigma` (and re-evaluates
    the noise/no-noise branch via `lax.cond`) on every call, even though `sigma` is fixed
    for the lifetime of a simulation run. This factory takes `sigma` once, resolves both
    at build time, and returns a per-step callable whose hot path is just a matrix-vector
    product -- matching the factory pattern used for controllers/planners elsewhere in
    this codebase.

    Args:
        sigma (Array): measurement model covariance matrix, fixed for the lifetime of
            the returned sensor.

    Returns
    -------
        sensor (SensorCallable): callable with signature `(t, x, *, sigma=None, key=None,
            **kwargs) -> Array`. The `sigma` keyword is accepted only for interface
            compatibility with `SensorCallable`/the simulator's call convention -- it is
            ignored, and the covariance captured at construction time is always used.
    """
    dim = sigma.shape[0]
    has_noise = bool(jnp.trace(jnp.abs(sigma)) > 0)
    chol = jnp.linalg.cholesky(sigma) if has_noise else jnp.zeros(sigma.shape)

    def sensor(
        t: Time,
        x: Array,
        *,
        sigma: Optional[Array] = None,
        key: Optional[Key] = None,
        **kwargs: Any,
    ) -> Array:
        """Senses the state subject to additive, unbiased Gaussian noise with the
        covariance fixed at factory-build time.

        Args:
            t (float): time (sec), unused
            x (Array): state vector (ground truth)
            sigma: unused -- accepted for `SensorCallable` compatibility only; the
                covariance baked in by `unbiased_gaussian_noise_factory` is always used.
            key (Key): PRNG key

        Returns
        -------
            y (Array): measurement of full state vector
        """
        del t, sigma
        if key is None:
            key = random.PRNGKey(0)  # type: ignore[assignment]

        z = random.normal(key, shape=(dim,))
        sampled_random_vector = jnp.dot(chol, z).reshape(x.shape)

        return x + sampled_random_vector

    return sensor
