
import jax.numpy as jnp
import pytest
from cbfkit.optimization.quadratic_program.qp_solver_jaxopt import solve_with_details

def test_qp_solver_correct_shapes():
    # 2 variables
    P = jnp.eye(2)
    q = jnp.array([1.0, 1.0]) # 1D

    # Inequality Constraints: x <= 0
    G = jnp.eye(2)
    h = jnp.array([0.0, 0.0]) # 1D

    # Equality Constraints: x[0] = 0
    A = jnp.array([[1.0, 0.0]])
    b = jnp.array([0.0]) # 1D

    sol, status, _ = solve_with_details(P, q, G, h, A, b)
    # Just check it runs without error
    assert sol.shape == (2,)

def test_qp_solver_incorrect_q_shape():
    P = jnp.eye(2)
    q = jnp.array([[1.0], [1.0]]) # 2D Column vector (Wrong)
    G = jnp.eye(2)
    h = jnp.array([0.0, 0.0])

    with pytest.raises(ValueError, match="Linear cost 'f_vec' must be a 1D array"):
        solve_with_details(P, q, G, h)

def test_qp_solver_incorrect_h_shape():
    P = jnp.eye(2)
    q = jnp.array([1.0, 1.0])
    G = jnp.eye(2)
    h = jnp.array([[0.0], [0.0]]) # 2D Column vector (Wrong)

    with pytest.raises(ValueError, match="Inequality constraint bounds 'h_vec' must be a 1D array"):
        solve_with_details(P, q, G, h)

def test_qp_solver_incorrect_b_shape():
    P = jnp.eye(2)
    q = jnp.array([1.0, 1.0])
    G = jnp.eye(2)
    h = jnp.array([0.0, 0.0])
    A = jnp.array([[1.0, 0.0]])
    b = jnp.array([[0.0]]) # 2D Column vector (Wrong)

    with pytest.raises(ValueError, match="Equality constraint bounds 'b_vec' must be a 1D array"):
        solve_with_details(P, q, G, h, A, b)

def test_qp_solver_incorrect_P_shape():
    P = jnp.eye(2).reshape(2, 2, 1) # 3D (Wrong)
    q = jnp.array([1.0, 1.0])

    with pytest.raises(ValueError, match="Quadratic cost 'h_mat' must be a 2D array"):
        solve_with_details(P, q)

def test_qp_solver_incorrect_G_shape():
    P = jnp.eye(2)
    q = jnp.array([1.0, 1.0])
    G = jnp.eye(2).reshape(2, 2, 1) # 3D (Wrong)
    h = jnp.array([0.0, 0.0])

    with pytest.raises(ValueError, match="Inequality constraint matrix 'g_mat' must be a 2D array"):
        solve_with_details(P, q, G, h)

def test_qp_solver_incorrect_A_shape():
    P = jnp.eye(2)
    q = jnp.array([1.0, 1.0])
    G = jnp.eye(2)
    h = jnp.array([0.0, 0.0])
    A = jnp.array([[1.0, 0.0]]).reshape(1, 2, 1) # 3D (Wrong)
    b = jnp.array([0.0])

    with pytest.raises(ValueError, match="Equality constraint matrix 'a_mat' must be a 2D array"):
        solve_with_details(P, q, G, h, A, b)
