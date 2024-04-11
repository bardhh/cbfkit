import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from jax import random, Array, jit
from scipy.linalg import block_diag
from cvxpylayers.jax import CvxpyLayer


KEY = random.PRNGKey(758493)  # Random seed is explicit in JAX
EPS = 1e-1
INFEASIBLE_AREA = -1e3


def block_diag_matrix_from_vec(n_blocks: int) -> Array:
    block = jnp.array([1, -1])
    return jnp.array([block_diag(*([block] * n_blocks)).T])[0, :, :]


def interleave_arrays(a: Array, b: Array) -> Array:
    return jnp.ravel(jnp.column_stack((a, b)))


def stochastic_barrier_transform(h: Array) -> Array:
    return jnp.exp(-h)


def stochastic_jacobian_transform(h: Array, dhdx: Array) -> Array:
    return -jnp.exp(-h) * dhdx


def stochastic_hessian_transform(h: Array, dhdx: Array, d2hdx2: Array) -> Array:
    return jnp.exp(-h) * (jnp.matmul(dhdx[:, None], dhdx[None, :]) - d2hdx2)


def compute_enclosed_area(
    A: np.ndarray, b: np.ndarray, x_max: Array, num_samples: int = 100000
) -> float:
    """Computes the area of a polytope using Monte Carlo sampling."""
    _, m = A.shape

    # Generate random points within a unit square
    points = random.uniform(KEY, shape=(num_samples, m)) - 0.5
    points = 2 * jnp.multiply(points, x_max)

    # Check if the points satisfy the inequality constraints
    satisfied = jnp.all(A @ points.T <= b.reshape(-1, 1), axis=0)
    # satisfied = A @ points.T <= b.reshape(-1, 1)

    # Compute the estimated enclosed area as the ratio of satisfied points to total points
    enclosed_area = jnp.sum(satisfied) / num_samples

    # Scale the estimated area by the area of the unit square
    enclosed_area *= jnp.prod(jnp.max(points, axis=0) - jnp.min(points, axis=0))

    if enclosed_area < EPS:
        enclosed_area = INFEASIBLE_AREA

    return float(enclosed_area)


def polytope_volume_ellipse_layer(n_decision_vars: int, n_affine_constraints: int):
    """Formulates the convex program for computing the inscribing ellipse of a polytope
    and returns the corresponding jax layer. The problem is

    min log(det(B))
    s.t.
    ||B*ai|| + ai*d <= bi

    and is based on an affine transformation of a circle:
    E = {Bu + d | ||u|| <= 1}

    Arguments:
        n_decision_vars: number of decision variables in the qp
        n_affine_constraints: number of affine constraints whose intersection forms the polytope

    Returns:
        CvxpyLayer: used to compute the sensitivity of the volume wrt the parameters



    """
    # Formulate and solve the Ellipse problem
    B = cp.Variable((n_decision_vars, n_decision_vars), symmetric=True)
    d = cp.Variable((n_decision_vars, 1))
    A = cp.Parameter((n_affine_constraints, n_decision_vars))
    b = cp.Parameter((n_affine_constraints, 1))
    objective = cp.Maximize(cp.log_det(B))
    constraints = [
        cp.norm(B @ A[ii, :]) + A[ii, :] @ d <= b[ii, 0] for ii in range(n_affine_constraints)
    ]
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[A, b], variables=[B, d])


@jit
def approximate_inscribing_ellipse_volume(B: Array) -> float:
    """Computes value proportional to the volume of the incribing ellipse of
    the polytope in question.

    Arguments:
        B: multiplying matrix in affine transformation of circle

    Returns:
        log(det(B))

    """
    return jnp.log(jnp.linalg.det(B))


###############################################################################
############################ Predictive CBF Utils #############################
###############################################################################


def mpc_layer(problem: cp.Problem) -> CvxpyLayer:
    """Formulates the convex program for computing the inscribing ellipse of a polytope
    and returns the corresponding jax layer. The problem is

    min log(det(B))
    s.t.
    ||B*ai|| + ai*d <= bi

    and is based on an affine transformation of a circle:
    E = {Bu + d | ||u|| <= 1}

    Arguments:
        n_decision_vars: number of decision variables in the qp
        n_affine_constraints: number of affine constraints whose intersection forms the polytope

    Returns:
        CvxpyLayer: used to compute the sensitivity of the volume wrt the parameters



    """
    # Formulate and solve the Ellipse problem
    B = cp.Variable((n_decision_vars, n_decision_vars), symmetric=True)
    d = cp.Variable((n_decision_vars, 1))
    A = cp.Parameter((n_affine_constraints, n_decision_vars))
    b = cp.Parameter((n_affine_constraints, 1))
    objective = cp.Maximize(cp.log_det(B))
    constraints = [
        cp.norm(B @ A[ii, :]) + A[ii, :] @ d <= b[ii, 0] for ii in range(n_affine_constraints)
    ]
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[A, b], variables=[B, d])


if __name__ == "__main__":
    A = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]])
    b = jnp.array([10, 10, 2, 2, -11.9])
    xmax = jnp.array([10, 2])
    area = compute_enclosed_area(A, b, xmax)
    print(area)
