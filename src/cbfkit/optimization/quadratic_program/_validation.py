"""Static QP shape checks shared by native and JAX solver adapters."""

from jax import Array


def validate_qp_shapes(
    h_mat: Array,
    f_vec: Array,
    g_mat: Array | None = None,
    h_vec: Array | None = None,
    a_mat: Array | None = None,
    b_vec: Array | None = None,
) -> None:
    """Reject incomplete constraints and incompatible shapes, including under JIT.

    These checks use only static shapes, never traced array values. Numerical
    convergence remains the backend's responsibility.
    """
    if f_vec.ndim != 1 or f_vec.shape[0] == 0:
        raise ValueError(
            "Linear cost 'f_vec' must be a 1D array of shape (n_vars,) with n_vars > 0."
        )
    n = f_vec.shape[0]
    if h_mat.shape != (n, n):
        raise ValueError(
            f"Quadratic cost 'h_mat' must be a 2D array of shape ({n}, {n}), got {h_mat.shape}."
        )
    for matrix, vector, matrix_name, vector_name, kind in (
        (g_mat, h_vec, "g_mat", "h_vec", "Inequality"),
        (a_mat, b_vec, "a_mat", "b_vec", "Equality"),
    ):
        if (matrix is None) != (vector is None):
            raise ValueError(f"'{matrix_name}' and '{vector_name}' must be supplied together.")
        if matrix is not None and vector is not None:
            if matrix.ndim != 2 or matrix.shape[1] != n:
                raise ValueError(
                    f"{kind} constraint matrix '{matrix_name}' must be a 2D array "
                    f"of shape (n_constraints, {n}), got {matrix.shape}."
                )
            if vector.shape != (matrix.shape[0],):
                raise ValueError(
                    f"{kind} constraint bounds '{vector_name}' must be a 1D array "
                    f"of shape ({matrix.shape[0]},), got {vector.shape}."
                )
