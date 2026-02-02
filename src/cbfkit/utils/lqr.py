"""LQR controller utility."""

import numpy as np
from scipy.linalg import solve_continuous_are


def lqr(A, B, Q, R):
    """Solve the continuous time LQR controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u

    Args:
        A (array_like): Dynamics matrix
        B (array_like): Input matrix
        Q (array_like): State cost matrix
        R (array_like): Input cost matrix

    Returns:
        K (ndarray): State feedback gain
        S (ndarray): Solution to Riccati equation
        E (None): Eigenvalues of the closed loop system (not computed)
    """
    # Ensure numpy arrays
    A_np = np.array(A)
    B_np = np.array(B)
    Q_np = np.array(Q)
    R_np = np.array(R)

    # Solve Algebraic Riccati Equation
    S = solve_continuous_are(A_np, B_np, Q_np, R_np)

    # Compute gain K = R^-1 B^T S
    K = np.linalg.inv(R_np) @ B_np.T @ S

    return K, S, None
