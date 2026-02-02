import jax.numpy as jnp
import pytest
from cbfkit.certificates.rectifiers import compute_function_list

def test_time_varying_rectification():
    """
    Test that compute_function_list correctly handles time-varying functions
    by including the partial time derivative in the Lie derivative computation.
    """
    # System: Double Integrator
    # x = [x1, x2]
    # dot_x1 = x2
    # dot_x2 = u
    # f(x) = [x2, 0]
    # g(x) = [0, 1]

    def dynamics(x):
        return jnp.array([x[1], 0.0]), jnp.array([[0.0], [1.0]])

    # Constraint: x1 >= t  => h = x1 - t >= 0
    # Augmented state for rectification: X = [x1, x2, t]
    def h(X):
        return X[0] - X[2]

    # Expected derivatives:
    # h = x1 - t
    # dot_h = L_f h + dh/dt
    # L_f h = [1, 0] * [x2, 0] = x2
    # dh/dt = -1
    # dot_h = x2 - 1

    # Run compute_function_list
    # state_dim passed is 3 (2 states + 1 time)
    funcs = compute_function_list(h, dynamics, 3)

    # Check the second function (index 1) which should be dot_h
    # funcs[0] is h, funcs[1] is dot_h
    h_func = funcs[0]
    dot_h_func = funcs[1]

    # Test point: x1=0, x2=10, t=5
    X_test = jnp.array([0.0, 10.0, 5.0])

    val_h = h_func(X_test) # 0 - 5 = -5
    val_dot_h = dot_h_func(X_test) # Expected: 10 - 1 = 9. Current code likely gives 10.

    print(f"h(X) = {val_h}")
    print(f"dot_h(X) = {val_dot_h}")

    # We use a loose tolerance because random sampling is involved in compute_function_list (to check relative degree),
    # but the derivative calculation itself is deterministic/symbolic via AD.
    assert jnp.isclose(val_dot_h, 9.0), f"Expected dot_h=9.0 (x2-1), got {val_dot_h}"
