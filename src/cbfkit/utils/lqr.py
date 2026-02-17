"""LQR utility module."""

import jax.numpy as jnp
import numpy as np
from jax import Array


def compute_lqr_gain(A: Array, B: Array, Q: Array, R: Array) -> Array:
    """Computes the LQR gain matrix K for the system dx/dt = Ax + Bu.

    Minimizes the cost function:
        J = integral(x.T * Q * x + u.T * R * u) dt

    The optimal control is u = -K * x.

    Args:
        A (Array): System matrix
        B (Array): Input matrix
        Q (Array): State cost matrix
        R (Array): Input cost matrix

    Returns
    -------
        Array: Gain matrix K
    """
    try:
        from scipy.linalg import solve_continuous_are
    except ImportError as e:
        raise ImportError(
            "To use 'compute_lqr_gain', please install the 'scipy' extra: "
            "pip install 'cbfkit[scipy]'"
        ) from e

    # Convert JAX arrays to NumPy arrays for SciPy
    A_np = np.array(A)
    B_np = np.array(B)
    Q_np = np.array(Q)
    R_np = np.array(R)

    # Solve Continuous Algebraic Riccati Equation (CARE)
    # A.T * P + P * A - P * B * R^-1 * B.T * P + Q = 0
    P = solve_continuous_are(A_np, B_np, Q_np, R_np)

    # Compute gain K = R^-1 * B.T * P
    # Note: solve_continuous_are assumes u = -Kx form implicitly for stability,
    # but strictly returns P.
    # K = inv(R) * B.T * P
    R_inv = np.linalg.inv(R_np)
    K_np = R_inv @ B_np.T @ P

    return jnp.array(K_np)
