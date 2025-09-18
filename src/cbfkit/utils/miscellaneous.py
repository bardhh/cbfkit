import jax.numpy as jnp

from cbfkit.utils.matrix_vector_operations import (
    block_diag_matrix,
    hat,
    normalize,
    vee,
)

__all__ = ["normalize", "hat", "vee", "block_diag_matrix", "tanh_sigmoid_func"]


def tanh_sigmoid_func(x: float, xbar: float):
    """Compute the value of the hyperbolic tangent sigmoid function.

    Args:
        x (float): argument to sigmoid.
        xbar (float): maximum value of argument.

    Returns:
        float: Result of the smooth saturation function.
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
