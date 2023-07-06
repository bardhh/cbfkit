import jax.numpy as jnp
from scipy.linalg import block_diag
from sympy import Matrix, symbols


def block_diag_matrix(n_blocks):
    block = jnp.array([1, -1])
    return jnp.array([block_diag(*([block] * n_blocks)).T])[0, :, :]


def interleave_arrays(a, b):
    return jnp.ravel(jnp.column_stack((a, b)))


def stochastic_barrier_transform(h):
    return jnp.exp(-h)


def stochastic_jacobian_transform(h, dhdx):
    return -jnp.exp(-h) * dhdx


def stochastic_hessian_transform(h, dhdx, d2hdx2):
    return jnp.exp(-h) * (jnp.matmul(dhdx[:, None], dhdx[None, :]) - d2hdx2)
