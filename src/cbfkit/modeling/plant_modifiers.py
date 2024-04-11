"""
#! docstring
"""

import jax.numpy as jnp
from jax import jit, Array
from typing import Optional, Union, Callable
from cbfkit.utils.user_types import DynamicsCallable
from .additive_disturbances import (
    generate_stochastic_perturbation,
    generate_bounded_perturbation,
    generate_affine_perturbation,
)


def sde_plant(
    plant: Callable[[Optional[Union[Callable[[Array], Array], None]]], DynamicsCallable],
    sigma: Callable[[Array], Array],
    dt: float,
) -> DynamicsCallable:
    """Plant model in stochastic differential equation (SDE) form.

    dx = (f(x) + g(x)u)dt + s(x)dw

    Args:
        sigma (Callable[Array, [Array]]): diffusion function in SDE
        dt (float): simulation time interval (in sec)

    Returns:
        plant (DynamicsCallable): plant model with additive, stochastic perturbation
    """

    return plant(generate_stochastic_perturbation(sigma, dt))


def ode_bounded_perturbation_plant(
    plant: Callable[[Optional[Union[Callable[[Array], Array], None]]], DynamicsCallable],
    perturbation: Callable[[Array], Array],
) -> DynamicsCallable:
    """Plant model in ordinary differential equation (ODE) form subject to
    an additive, bounded perturbation (function of the state).

    dx/dt = f(x) + g(x)u + p(x)

    Args:
        perturbation (Callable[Array, [Array]]): additive perturbation to nominal dynamics

    Returns:
        plant (DynamicsCallable): plant model with additive, bounded perturbation
    """

    return plant(generate_bounded_perturbation(perturbation))


def ode_affine_perturbation_plant(
    plant: Callable[[Optional[Union[Callable[[Array], Array], None]]], DynamicsCallable],
    regressor: Callable[[Array], Array],
    parameter_vector: Array,
) -> DynamicsCallable:
    """Plant model in ordinary differential equation (ODE) form subject to
    an additive, parameter-affine perturbation (function of the state).

    dx/dt = f(x) + g(x)u + d(x) * p

    Args:
        regressor (Callable[Array, [Array]]): regressor matrix function
        parameter_vector (Array): vector of affine parameters multiplying regressor

    Returns:
        plant (DynamicsCallable): plant model with additive, parameter-affine perturbation
    """

    return plant(generate_affine_perturbation(regressor, parameter_vector))


def ode_plant(
    plant: Callable[[Optional[Union[Callable[[Array], Array], None]]], DynamicsCallable]
) -> DynamicsCallable:
    """Nominal plant model in ordinary differential equation (ODE) form.

    dx/dt = f(x) + g(x)u

    Args:
        None

    Returns:
        plant (DynamicsCallable): nominal plant model
    """
    return plant()
