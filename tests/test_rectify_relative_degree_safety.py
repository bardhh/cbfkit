
import pytest
import jax.numpy as jnp
from cbfkit.certificates import rectify_relative_degree

def test_rectify_relative_degree_uncontrollable():
    """
    Test that rectify_relative_degree raises ValueError for an uncontrollable system
    instead of entering infinite recursion.
    """

    # Dynamics: x_dot = x, but g(x) = 0 (no control)
    def dynamics(x):
        return x, jnp.zeros((2, 1))

    # Barrier: h(x) = x[0]
    # Lg h = 0, Lg Lf h = 0, ...
    def h(x):
        return x[0]

    state_dim = 2

    # Expect ValueError because relative degree cannot be found within state dimension limits
    with pytest.raises(ValueError, match="Relative degree undefined or exceeds system dimension"):
        rectify_relative_degree(
            function=h,
            system_dynamics=dynamics,
            state_dim=state_dim,
            form="exponential"
        )

if __name__ == "__main__":
    test_rectify_relative_degree_uncontrollable()
