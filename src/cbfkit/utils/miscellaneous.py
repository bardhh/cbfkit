from cbfkit.utils.matrix_vector_operations import block_diag_matrix, hat, normalize, vee
from cbfkit.utils.real_functions import tanh_sigmoid as tanh_sigmoid_func

__all__ = ["normalize", "hat", "vee", "block_diag_matrix", "tanh_sigmoid_func"]


if __name__ == "__main__":
    x = jnp.array([1, 1, 0, 1])
    y = jnp.array([[1, 2], [3, 4]])
    print(normalize(x, axis=0))
    print(normalize(y))
