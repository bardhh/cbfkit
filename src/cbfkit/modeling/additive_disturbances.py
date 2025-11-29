"""
#! docstring
"""

from typing import Callable

import jax.numpy as jnp
from jax import Array, jit, random

from cbfkit.utils.user_types import Key, PerturbationCallable, PerturbationCallableReturns


def generate_stochastic_perturbation(
    sigma: Callable[[Array], Array], dt: float
) -> PerturbationCallable:
    """Generates additive, stochastic perturbation function given sigma (diffusion function).

    Args:
        sigma (Callable[Array, [Array]]): diffusion function of the state
        dt (float): simulation time interval (sec)

    Returns:
        Callable[Array, [Array]]: function to compute stochastic perturbation based on state
    """

    def stochastic_perturbation(
        x: Array, _u: Array, _f: Array, _g: Array
    ) -> PerturbationCallableReturns:
        """Computes value of stochastic perturbation.

        Args:
            x (Array): state vector

        Returns:
            Array: value
        """
        nonlocal sigma
        sigma_x = sigma(x)
        return generate_compute(sigma_x)

    def generate_compute(sigma_x: Array) -> PerturbationCallableReturns:
        """"""

        def compute(subkey: Key) -> Array:
            """TO DO"""
            nonlocal dt
            dw = random.normal(subkey, shape=(sigma_x.shape[1],)) * jnp.sqrt(dt)
            return jnp.matmul(sigma_x, dw) / dt

        return compute

    return stochastic_perturbation


def generate_bounded_perturbation(
    perturbation: Callable[[Array], Array],
) -> PerturbationCallable:
    """Generates additive, bounded perturbation to system dynamics.

    Args:
        perturbation (Callable[Array, [Array]]): bounded perturbation of the form ||d(x)|| <= D

    Returns:
        Callable[Array, [Array]]: function to compute additive perturbation
    """

    def bounded_perturbation(
        x: Array, _u: Array, _f: Array, _g: Array
    ) -> PerturbationCallableReturns:
        """Computes value of stochastic perturbation.

        Args:
            x (Array): state vector

        Returns:
            Array: value
        """
        nonlocal perturbation
        perturb = perturbation(x)
        return generate_compute(perturb)

    def generate_compute(perturbation_x: Array) -> PerturbationCallableReturns:
        """"""

        @jit
        def compute(_subkey: Key) -> Array:
            """TO DO"""
            nonlocal perturbation_x
            return perturbation_x

        return compute

    return bounded_perturbation


def generate_affine_perturbation(
    regressor: Callable[[Array], Array], parameter_vector: Array
) -> PerturbationCallable:
    """Generates additive, bounded perturbation to system dynamics.

    Args:
        regressor (Callable[Array, [Array]]): bounded perturbation of the form ||d(x)|| <= D
        parameter_vector (Array): vector of unknown (or known) parameters, possibly belonging
            to a bounded set

    Returns:
        Callable[Array, [Array]]: function to compute additive perturbation
    """

    def affine_perturbation(
        x: Array, _u: Array, _f: Array, _g: Array
    ) -> PerturbationCallableReturns:
        """Computes value of an additive, parameter-affine perturbation of the form Delta(x) * theta.

        Args:
            x (Array): state vector

        Returns:
            Array: value
        """
        nonlocal regressor, parameter_vector
        perturb = jnp.matmul(regressor(x), parameter_vector)
        return generate_compute(perturb)

    def generate_compute(perturbation_x: Array) -> PerturbationCallableReturns:
        """"""

        @jit
        def compute(_subkey: Key) -> Array:
            """TO DO"""
            nonlocal perturbation_x
            return perturbation_x

        return compute

    return affine_perturbation
