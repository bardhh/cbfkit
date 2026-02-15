
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

    # Updated behavior: State is frozen at last valid value, not NaN
    assert not jnp.any(jnp.isnan(results.states))
    assert jnp.allclose(results.states, x0)

    # Check captured stderr for error/warning messages (rich Console writes to stderr)
    captured = capsys.readouterr()

    # "Simulation failed due to NaNs" is only printed if NaNs are present in the trajectory.
    # Since we prevent them, this is NOT printed.
    # assert "Simulation failed due to NaNs" in captured.err

    # Controller error MUST be flagged with correct status (-10)
    assert "CONTROLLER ERROR: INTEGRATION_ERROR" in captured.err

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

    # JIT returns full size array.
    # Sentinel fix: NaNs should be prevented (frozen state), but error flagged.
    assert results.states.shape[0] == 10
    assert not jnp.any(jnp.isnan(results.states))

    # Verify trajectory is frozen
    assert jnp.allclose(results.states, x0)

    captured = capsys.readouterr()
    # "Sentinel: Simulation failed due to NaNs" is only printed if NaNs are present in the final trajectory.
    # Since we prevent them, this message is NOT printed.

    # However, controller error must be flagged (rich Console writes to stderr).
    assert "Simulation stopped early due to controller error" in captured.err
