"""
matrix_vector_operations
================

This file contains miscellaneous helper functions for performing
operations involving matrices and vectors.

Functions
---------
normalize(a, axis, order): normalizes a matrix or vector to be of (order) norm 1
hat(v): performs the hat operation (skew-symmetric matrix) on a vector
vee(M): performs the V operation on a matrix
block_diag_matrix(A, B): creates a block diagonal matrix of the form [A, 0; 0, B]

Notes
-----
This module is, and will continue to be, a work in progress with more functions
added as necessary.

Examples
--------
>>> from matrix_vector_operations import *
>>> import jax.numpy as jnp
>>> a = jnp.array([1, 2, 3])
>>> a_normalized = normalize(a)
>>> a_hat = hat(a)
>>> a = = vee(a_hat)
>>> double_a = block_diag_matrix(a_hat, a_hat)

"""
import jax.numpy as jnp
from jax import Array, jit, lax


@jit
def normalize(a: Array, axis: int = -1, order: int = 2) -> Array:
    """Normalizes given array (matrix or vector).

    Args:
        a (Array): matrix or vector to normalize
        axis (int, optional): axis of normalization. Defaults to -1.
        order (int, optional): norm to use (2 = Euclidean). Defaults to 2.

    Returns:
        Array: normalized array
    """

    l2 = jnp.atleast_1d(jnp.linalg.norm(a, order, axis))
    mask = l2 == 0
    modified_l2 = lax.cond(mask.any(), lambda x: jnp.where(mask, 1, x), lambda x: x, l2)

    return (a / jnp.expand_dims(modified_l2, axis)).reshape(a.shape)


@jit
def hat(v: Array) -> Array:
    """Performs the hat operation on the given array (vector), e.g.,

    v_hat = [
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ]

    Args:
        v (Array): array on which to perform hat operation

    Returns:
        Array: hat matrix
    """
    assert len(v.shape) == 1 or (len(v.shape) < 2 and v.shape[-1] == 1)
    return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


@jit
def vee(mat: Array) -> Array:
    """Performs the V operation on a 3x3 matrix, e.g.,

    mat_vee = [
        (mat[2, 1] - mat[1, 2]) / 2,
        (mat[0, 2] - mat[2, 0]) / 2,
        (mat[1, 0] - mat[0, 1]) / 2,
    ]

    Args:
        mat (Array): 3x3 matrix

    Returns:
        Array: resulting vector
    """
    assert mat.shape == (3, 3)
    m1 = (mat[2, 1] - mat[1, 2]) / 2
    m2 = (mat[0, 2] - mat[2, 0]) / 2
    m3 = (mat[1, 0] - mat[0, 1]) / 2
    return jnp.array([m1, m2, m3])


@jit
def block_diag_matrix(mat1: Array, mat2: Array) -> Array:
    """Creates new block diagonal matrix block_mat = [[mat1, 0], [0, mat2]] from
    two matrices mat1, mat2.

    Args:
        mat1 (Array): first matrix (top block)
        mat2 (Array): second matrix (bottom block)

    Returns:
        Array: blocked matrix
    """
    assert len(mat1.shape) == 2 and len(mat2.shape) == 2
    m, n = mat1.shape
    p, q = mat2.shape

    block_mat = jnp.zeros((m + p, n + q), dtype=mat1.dtype)
    block_mat = block_mat.at[:m, :n].set(mat1)
    block_mat = block_mat.at[m:, n:].set(mat2)

    return block_mat


@jit
def invert_array(a):
    # Check if the array is already 2D
    if a.ndim == 1:
        # Reshape the vector into a 2D array (1x1 matrix)
        a_matrix = a.reshape(
            (1, -1)
        )  # -1 automatically computes the size of the remaining dimension
    else:
        # If it's already 2D, use it as is
        a_matrix = a

    # Compute the inverse
    a_inverse = jnp.linalg.inv(a_matrix)

    return a_inverse
