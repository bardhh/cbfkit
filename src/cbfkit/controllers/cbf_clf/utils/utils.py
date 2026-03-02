import jax.numpy as jnp
from jax import Array, random

KEY = random.PRNGKey(758493)  # Random seed is explicit in JAX
EPS = 1e-1
INFEASIBLE_AREA = -1e3


def block_diag_matrix_from_vec(n_blocks: int) -> Array:
    """Constructs a block diagonal matrix for box constraints.

    Creates a matrix A of shape (2*N, N) where each column i corresponds to input u_i,
    and has entries 1 at row 2i and -1 at row 2i+1.
    This corresponds to constraints u_i <= b and -u_i <= b.

    Args:
        n_blocks (int): Number of inputs (N).

    Returns:
        Array: The constraint matrix A.
    """
    # Use pure JAX implementation to avoid scipy dependency and implicit conversion issues.
    # We want columns [1, -1]^T on the diagonal.
    block = jnp.array([[1.0], [-1.0]])
    return jnp.kron(jnp.eye(n_blocks), block)


def interleave_arrays(a: Array, b: Array) -> Array:
    return jnp.ravel(jnp.column_stack((a, b)))


def stochastic_barrier_transform(h: Array) -> Array:
    return jnp.exp(-h)


def stochastic_jacobian_transform(h: Array, dhdx: Array) -> Array:
    return -jnp.exp(-h) * dhdx


def stochastic_hessian_transform(h: Array, dhdx: Array, d2hdx2: Array) -> Array:
    return jnp.exp(-h) * (jnp.matmul(dhdx[:, None], dhdx[None, :]) - d2hdx2)
