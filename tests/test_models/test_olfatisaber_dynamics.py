import pytest
import jax.numpy as jnp
from cbfkit.systems.unicycle.models.olfatisaber2002approximate.dynamics import approx_unicycle_dynamics

def test_olfatisaber_dynamics_state_shape():
    """
    Verifies that the dynamics function accepts a 3-element state (x, y, theta)
    and rejects a 4-element state (x, y, theta, l), consistent with the updated docstring.
    """
    dynamics_func = approx_unicycle_dynamics(lam=0.5)

    # Test with valid 3-element state
    state_3 = jnp.array([1.0, 2.0, jnp.pi/4])
    f, g = dynamics_func(state_3)

    assert f.shape == (3,)
    assert g.shape == (3, 2)

    # Check values for consistency
    # f should be [0, 0, 0]
    assert jnp.allclose(f, jnp.zeros(3))

    # Test with invalid 4-element state
    state_4 = jnp.array([1.0, 2.0, jnp.pi/4, 0.5])

    # The function unpacks with `_, _, theta = state`
    # This should raise a ValueError when state has 4 elements.
    with pytest.raises(ValueError, match="too many values to unpack"):
        _ = dynamics_func(state_4)
