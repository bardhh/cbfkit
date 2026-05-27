"""Outward radial 'task-drive' nominal controller (pushes the robot toward the boundary)."""
import jax.numpy as jnp
from jax import Array

from cbfkit.controllers.utils import setup_nominal_controller


def outward_drive(v_max: float):
    """NominalControllerCallable driving u = v_max * x / ||x|| (radially outward)."""

    def u_nom(t: float, x: Array) -> Array:
        return v_max * x / (jnp.linalg.norm(x) + 1e-6)

    return setup_nominal_controller(u_nom)
