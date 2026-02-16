
import pytest
import jax.numpy as jnp
from cbfkit.systems.unicycle.models.accel_unicycle import plant
from cbfkit.simulation import simulator
from cbfkit.integration import forward_euler

def test_simulator_shape_mismatch_column_vector():
    """Test that simulator.execute raises ValueError when x0 is a column vector (N, 1)."""
    dynamics = plant()
    # Accel unicycle has 4 states
    # Passing (4, 1) causes dynamics evaluation failure (JAX error wrapped in ValueError)
    x0 = jnp.array([[0.0], [0.0], [1.0], [0.0]])

    with pytest.raises(ValueError, match="Dynamics evaluation failed for initial state 'x0'"):
        simulator.execute(
            x0=x0,
            dt=0.1,
            num_steps=10,
            dynamics=dynamics,
            integrator=forward_euler,
            planner=None,
            nominal_controller=None,
        )

def test_simulator_shape_mismatch_wrong_dimension():
    """Test that simulator.execute raises ValueError when x0 has wrong number of states."""
    dynamics = plant()
    # Accel unicycle has 4 states
    # Passing 3 states should trigger wrong dimension error
    x0 = jnp.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="Shape mismatch: Initial state 'x0' has shape"):
        simulator.execute(
            x0=x0,
            dt=0.1,
            num_steps=10,
            dynamics=dynamics,
            integrator=forward_euler,
            planner=None,
            nominal_controller=None,
        )

def test_simulator_valid_input():
    """Test that simulator.execute works with correct shape."""
    dynamics = plant()
    x0 = jnp.array([0.0, 0.0, 1.0, 0.0])

    # Should not raise
    simulator.execute(
        x0=x0,
        dt=0.1,
        num_steps=10,
        dynamics=dynamics,
        integrator=forward_euler,
        planner=None,
        nominal_controller=lambda t, x, k, r: (jnp.zeros(2), {}),
        use_jit=True,
    )
