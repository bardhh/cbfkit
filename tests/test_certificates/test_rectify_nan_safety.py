
import pytest
import jax.numpy as jnp
from cbfkit.certificates import rectify_relative_degree

def test_rectify_relative_degree_detects_nan_dynamics():
    """
    Verifies that rectify_relative_degree raises a ValueError when the underlying
    dynamics function returns NaNs during the relative degree analysis.
    """

    def broken_dynamics(x):
        # Returns NaNs to simulate a numerical failure or bad definition
        return jnp.full_like(x, jnp.nan), jnp.full((x.shape[0], 1), jnp.nan)

    def h(x):
        # Relative degree > 1 so it requires recursion
        return x[0]

    # Analysis should fail immediately
    with pytest.raises(ValueError, match="Encountered NaN during relative degree verification"):
        rectify_relative_degree(h, broken_dynamics, state_dim=2)

def test_rectify_relative_degree_detects_nan_gradients():
    """
    Verifies that rectify_relative_degree raises a ValueError when the constraint
    function gradient evaluation results in NaNs.
    """

    def correct_dynamics(x):
         return jnp.array([x[1], 0.0]), jnp.array([[0.0], [1.0]])

    def nan_gradient_h(x):
        # Function that produces NaN gradients (e.g. sqrt of negative, or explicitly NaN)
        # Using explicit NaN to be sure
        return jnp.nan * x[0]

    with pytest.raises(ValueError, match="Encountered NaN during relative degree verification"):
        rectify_relative_degree(nan_gradient_h, correct_dynamics, state_dim=2)
