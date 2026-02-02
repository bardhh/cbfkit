
import jax.numpy as jnp
import pytest
from cbfkit.simulation.simulator import execute
from cbfkit.utils.user_types import ControllerData

def nan_dynamics(x):
    # Returns NaNs
    return jnp.full((2,), jnp.nan), jnp.full((2, 1), jnp.nan)

def simple_integrator(x, vector_field, dt):
    return x + vector_field(x) * dt

def dummy_nominal_controller(t, x, key, data):
    return jnp.zeros((1,)), {}

def test_nan_detection_python_loop(capsys):
    """Test that NaNs are detected in the Python simulation loop."""
    x0 = jnp.array([1.0, 1.0])
    dt = 0.1
    num_steps = 10

    results = execute(
        x0=x0,
        dt=dt,
        num_steps=num_steps,
        dynamics=nan_dynamics,
        integrator=simple_integrator,
        nominal_controller=dummy_nominal_controller,
        verbose=True,
        use_jit=False
    )

    # Check that simulation stopped early (result states should be padded or short?
    # simulator returns stacked array of whatever was collected.
    # If it stopped at step 0, it has 1 element (or 0?).
    # simulator loop: yields step_data.
    # if error at step 0:
    #   yield step_data (NaN)
    #   break
    # So 1 element.

    assert results.states.shape[0] == 1
    assert jnp.any(jnp.isnan(results.states))

    # Check captured stdout for Sentinel warning
    captured = capsys.readouterr()
    assert "Sentinel: Simulation failed due to NaNs" in captured.out
    assert "CONTROLLER ERROR: NAN_DETECTED" in captured.out

def test_nan_detection_jit_loop(capsys):
    """Test that NaNs are detected in the JIT simulation loop."""
    x0 = jnp.array([1.0, 1.0])
    dt = 0.1
    num_steps = 10

    results = execute(
        x0=x0,
        dt=dt,
        num_steps=num_steps,
        dynamics=nan_dynamics,
        integrator=simple_integrator,
        nominal_controller=dummy_nominal_controller,
        verbose=True,
        use_jit=True
    )

    # JIT returns full size array but logic should flag error
    assert results.states.shape[0] == 10
    assert jnp.any(jnp.isnan(results.states))

    captured = capsys.readouterr()
    assert "Sentinel: Simulation failed due to NaNs" in captured.out
    # JIT loop might not print "CONTROLLER ERROR: NAN_DETECTED" because error_data is None?
    # In my verify run, I saw:
    # Sentinel: Simulation failed due to NaNs in state trajectory.
    # Warning: Simulation stopped early due to controller error at step 0.

    assert "Simulation stopped early due to controller error" in captured.out
