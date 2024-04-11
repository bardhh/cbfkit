import jax.numpy as jnp
from jax import jit, Array
from typing import Optional, Union, Callable
from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns
from cbfkit.modeling.additive_disturbances import (
    generate_stochastic_perturbation,
    generate_bounded_perturbation,
)

g_accel = 9.81


def sde_plant(sigma: Callable[[Array], Array], dt: float) -> DynamicsCallable:
    """Plant model in stochastic differential equation (SDE) form.

    dx = (f(x) + g(x)u)dt + s(x)dw

    The nominal model characterized by this function may be found in Beard et al. 2014
    "Fixed Wing UAV Path Following in Wind with Input Constraints".

    States are the following:
        x: x-position in inertial frame
        y: y-position in inertial frame
        z: z-position in inertial frame
        v: airspeed
        psi: yaw angle
        gamma: flight path angle

    Control inputs are the following:
        a: rate of change of airspeed
        omega: rate of change of gamma
        tan(phi): tangent of the roll angle

    Args:
        sigma (Callable[Array, [Array]]): diffusion function in SDE

    Returns:
        plant (DynamicsCallable): plant model with additive, stochastic perturbation
    """
    return plant(generate_stochastic_perturbation(sigma, dt))


def ode_perturbed_plant(perturbation: Callable[[Array], Array]) -> DynamicsCallable:
    """Plant model in ordinary differential equation (ODE) form subject to
    an additive, bounded perturbation (function of the state).

    dx/dt = f(x) + g(x)u + p(x)

    The nominal model characterized by this function may be found in Beard et al. 2014
    "Fixed Wing UAV Path Following in Wind with Input Constraints".

    States are the following:
        x: x-position in inertial frame
        y: y-position in inertial frame
        z: z-position in inertial frame
        v: airspeed
        psi: yaw angle
        gamma: flight path angle

    Control inputs are the following:
        a: rate of change of airspeed
        omega: rate of change of gamma
        tan(phi): tangent of the roll angle

    Args:
        perturbation (Callable[Array, [Array]]): additive perturbation to nominal dynamics

    Returns:
        plant (DynamicsCallable): plant model with additive, bounded perturbation
    """
    return plant(generate_bounded_perturbation(perturbation))


def ode_plant() -> DynamicsCallable:
    """Nominal plant model in ordinary differential equation (ODE) form.

    dx/dt = f(x) + g(x)u

    The model characterized by this function may be found in Beard et al. 2014
    "Fixed Wing UAV Path Following in Wind with Input Constraints".

    States are the following:
        x: x-position in inertial frame
        y: y-position in inertial frame
        z: z-position in inertial frame
        v: airspeed
        psi: yaw angle
        gamma: flight path angle

    Control inputs are the following:
        a: rate of change of airspeed
        omega: rate of change of gamma
        tan(phi): tangent of the roll angle

    Args:
        None

    Returns:
        plant (DynamicsCallable): nominal plant model
    """
    return plant()


def plant() -> DynamicsCallable:
    """
    Returns a function that represents the fixed-wing UAV dynamics,
    which computes the drift vector 'f', control matrix 'g', and a perturbation
    'p' (stochastic, deterministic, or None) based on the given state.

    The model characterized by this function may be found in Beard et al. 2014
    "Fixed Wing UAV Path Following in Wind with Input Constraints".

    States are the following:
        x: x-position in inertial frame
        y: y-position in inertial frame
        z: z-position in inertial frame
        v: airspeed
        psi: yaw angle
        gamma: flight path angle

    Control inputs are the following:
        a: rate of change of airspeed
        omega: rate of change of gamma
        tan(phi): tangent of the roll angle

    Args:
        perturbation (Optional, Callable / None): perturbation to nominal dynamics

    Returns:
        dynamics (DynamicsCallable): takes state as input and returns dynamics components
            f, g, and p of one of the following forms:

            Case 1: stochastic dynamics
            dx = (f(x) + g(x)u)dt + p(x)dw

            Case 2: deterministic dynamics w/ bounded disturbance
            dx/dt = f(x) + g(x)u + p(x)

            Case 3: perfect dynamics
            dx/dt = f(x) + g(x)u

    """

    @jit
    def kinematics(state: Array) -> DynamicsCallableReturns:
        """
        Computes the drift vector 'f', control matrix 'g', and perturbation 'p'
        based on the given state x, which consists of px, py, pz (positions in
        m), v (flight speed in m/s), gamma (flight path angle in rad), and psi
        (heading angle in rad).

        Args:
            state (Array): state vector

        Returns:
            f, g, p(state) (Tuple of Arrays or None): drift vector f, control matrix g, perturbation p(state)

        """

        _, _, _, v, psi, gamma = state
        f = jnp.array(
            [
                v * jnp.cos(psi) * jnp.cos(gamma),
                v * jnp.sin(psi) * jnp.cos(gamma),
                v * jnp.sin(gamma),
                0.0,
                0.0,
                0.0,
            ]
        )
        g = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, g_accel / v],
                [0.0, 1.0, 0.0],
            ]
        )

        return f, g

    return kinematics
