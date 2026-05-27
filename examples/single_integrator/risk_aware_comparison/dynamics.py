"""2D stochastic single integrator: dx = u dt + sigma dW (sigma = SIGMA * I_2)."""
import jax.numpy as jnp
from jax import Array

from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
from cbfkit.systems import single_integrator


def make_dynamics():
    """Control-affine (f, g): f = 0, g = I_2."""
    return single_integrator.two_dimensional_single_integrator()


def make_sigma(sigma_scalar: float):
    """Diffusion callable sigma(x) = sigma_scalar * I_2 (constant)."""
    mat = sigma_scalar * jnp.eye(2)

    def sigma(x: Array) -> Array:
        return mat

    return sigma


def make_perturbation(sigma_scalar: float, dt: float):
    """Brownian process-noise PerturbationCallable (scales by sqrt(dt) internally)."""
    return generate_stochastic_perturbation(sigma=make_sigma(sigma_scalar), dt=dt)
