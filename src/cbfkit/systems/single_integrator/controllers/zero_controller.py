import jax.numpy as jnp
from jax import Array

def zero_controller(dynamics=None, **kwargs):
    def controller(t: float, x: Array, key: Array = None, xd: Array = None):
        return jnp.zeros(2), {}
    return controller
