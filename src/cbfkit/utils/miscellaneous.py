import jax
import jax.numpy as jnp
from jax import Array, jit


def normalize(a, axis=-1, order=2):
    l2 = jnp.atleast_1d(jnp.linalg.norm(a, order, axis))
    mask = l2 == 0
    modified_l2 = jax.lax.cond(mask.any(), lambda x: jnp.where(mask, 1, x), lambda x: x, l2)

    return (a / jnp.expand_dims(modified_l2, axis)).reshape(a.shape)


def hat(v):
    return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def vee(M):
    m1 = (M[2, 1] - M[1, 2]) / 2
    m2 = (M[0, 2] - M[2, 0]) / 2
    m3 = (M[1, 0] - M[0, 1]) / 2
    return jnp.array([m1, m2, m3])


@jit
def block_diag_matrix(A: Array, B: Array) -> Array:
    """Creates new block diagonal matrix C = [[A, 0], [0, B]] from two matrices A, B.

    Args:
        A (Array): matrix 1
        B (Array): matrix 2

    Returns:
        Array: block diagonal matrix
    """
    m, n = A.shape
    p, q = B.shape

    C = jnp.zeros((m + p, n + q), dtype=A.dtype)
    C = C.at[:m, :n].set(A)
    C = C.at[m:, n:].set(B)

    return C


@jit
def tanh_sigmoid_func(x: float, xbar: float):
    """Computes the value of the hyperbolic tangent sigmoid function.

    Args:
        x (float): argument to sigmoid
        xbar (float): maximum value of argument

    Returns:
        float: result
    """
    k = 100
    return x * (1 / 2 + 1 / 2 * jnp.tanh(k * x)) + (xbar - x) * (
        1 / 2 + 1 / 2 * jnp.tanh(k * (x - xbar))
    )


if __name__ == "__main__":
    x = jnp.array([1, 1, 0, 1])
    y = jnp.array([[1, 2], [3, 4]])
    print(normalize(x, axis=0))
    print(normalize(y))
