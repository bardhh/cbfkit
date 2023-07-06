import jax.numpy as jnp
import numpy as np
from typing import Union, Tuple, Any, List
from kvxopt import matrix, solvers
from jax import Array

###############################################################################
############################ Currently Functional #############################
###############################################################################


def qp_solver(
    H: Array,
    f: Array,
    G: Union[Array, None] = None,
    h: Union[Array, None] = None,
    A: Union[Array, None] = None,
    b: Union[Array, None] = None,
) -> Tuple[Array, int]:
    """
    Solve a quadratic program using the cvxopt solver.

    Args:
    H: quadratic cost matrix.
    f: linear cost vector.
    A: linear constraint matrix.
    b: linear constraint vector.
    G: quadratic constraint matrix.
    h: quadratic constraint vector.

    Returns:
    sol['x']: Solution to the QP
    """
    # Use the cvxopt library to solve the quadratic program
    P = matrix(np.array(H))
    q = matrix(np.array(f))
    options = {"show_progress": False}

    if G is not None and h is not None:
        G = matrix(np.array(G))
        h = matrix(np.array(h))

    if A is not None and b is not None:
        A = matrix(np.array(A))
        b = matrix(np.array(b))

    try:
        sol: dict[str, Any] = solvers.qp(P, q, G=G, h=h, A=A, b=b, options=options)
    except:
        return jnp.array([0] * H.shape[0], dtype=int), 0

    success: bool = sol["status"] == "optimal"  # or (sol["status"] == "unknown")
    if not success:
        sol["x"] = 0 * sol["x"]

    return sol["x"], success
