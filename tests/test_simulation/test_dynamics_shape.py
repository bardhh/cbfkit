import pytest
import jax.numpy as jnp
from cbfkit.simulation import simulator
from cbfkit.integration import forward_euler

def test_dynamics_shape_validation():
    """Verifies that simulator.execute catches incorrect dynamics output shapes."""

    def bad_dynamics(x):
        # Returns column vector (2, 1) instead of (2,)
        f = jnp.array([[0.0], [0.0]])
        g = jnp.array([[1.0], [1.0]])
        return f, g

    x0 = jnp.array([0.0, 0.0])
    dt = 0.1
    num_steps = 2

    with pytest.raises(ValueError, match="Shape mismatch: Initial state 'x0' has shape"):
        simulator.execute(
            x0=x0,
            dt=dt,
            num_steps=num_steps,
            dynamics=bad_dynamics,
            integrator=forward_euler,
            use_jit=True
        )

def test_dynamics_g_shape_validation():
    """Verifies that simulator.execute catches incorrect G output shapes."""

    def bad_dynamics_g(x):
        f = jnp.array([0.0, 0.0])
        # Returns 1D array instead of 2D
        g = jnp.array([1.0, 1.0])
        return f, g

    x0 = jnp.array([0.0, 0.0])
    dt = 0.1
    num_steps = 2

    with pytest.raises(ValueError, match="Expected 2D array"):
        simulator.execute(
            x0=x0,
            dt=dt,
            num_steps=num_steps,
            dynamics=bad_dynamics_g,
            integrator=forward_euler,
            use_jit=True
        )
