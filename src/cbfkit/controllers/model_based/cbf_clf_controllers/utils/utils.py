import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from jax import random, Array, jit
from scipy.linalg import block_diag
from cvxpylayers.jax import CvxpyLayer


KEY = random.PRNGKey(758493)  # Random seed is explicit in JAX
EPS = 1e-1
INFEASIBLE_AREA = -1e3


def block_diag_matrix_from_vec(n_blocks: int) -> Array:
    block = jnp.array([1, -1])
    return jnp.array([block_diag(*([block] * n_blocks)).T])[0, :, :]


def interleave_arrays(a: Array, b: Array) -> Array:
    return jnp.ravel(jnp.column_stack((a, b)))


def stochastic_barrier_transform(h: Array) -> Array:
    return jnp.exp(-h)


def stochastic_jacobian_transform(h: Array, dhdx: Array) -> Array:
    return -jnp.exp(-h) * dhdx


def stochastic_hessian_transform(h: Array, dhdx: Array, d2hdx2: Array) -> Array:
    return jnp.exp(-h) * (jnp.matmul(dhdx[:, None], dhdx[None, :]) - d2hdx2)
