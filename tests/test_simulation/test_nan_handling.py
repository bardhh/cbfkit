"""
Tests for simulator robustness against NaN generation during integration.
Ensures that simulation stops cleanly and reports error -10 instead of propagating NaNs.
"""

import jax.numpy as jnp
from jax import jit
import pytest

from cbfkit.simulation.simulator import execute
from cbfkit.simulation.simulator_jit import INTEGRATION_NAN_ERROR
from cbfkit.integration.forward_euler import forward_euler
from cbfkit.utils.user_types import ControllerData

# 1D system where state x[0] represents "time"
def explosive_dynamics(x):
    # Dynamics: dx/dt = 1 (linear growth)
    # If x > 0.5, return NaN
    # We use jnp.where to conditionally return NaN
    # x[0] is scalar.
    val = jnp.where(x[0] > 0.5, jnp.nan, 1.0)

    # Ensure f is shape (1,)
    f = jnp.array([val])
    g = jnp.zeros((1, 1))

    return f, g

# Simple nominal controller required by simulator
def dummy_controller(t, x, key, ref):
    return jnp.zeros((1,)), {}

@pytest.mark.parametrize("use_jit", [False, True])
def test_integration_nan_handling(use_jit):
    """
    Verifies that the simulator detects NaNs during integration,
    stops early, and reports INTEGRATION_NAN_ERROR (-10).
    """
    dt = 0.1
    num_steps = 10 # Total time 1.0
    x0 = jnp.array([0.0])

    # Initialize controller_data with error_data=0 so JIT can track it.
    # Without this, error_data remains None in JIT trace.
    initial_c_data = ControllerData(error_data=jnp.array(0))

    # Run simulation
    results = execute(
        x0=x0,
        dt=dt,
        num_steps=num_steps,
        dynamics=explosive_dynamics,
        integrator=forward_euler,
        nominal_controller=dummy_controller,
        controller_data=initial_c_data,
        use_jit=use_jit,
        verbose=False
    )

    # 1. Safety Invariant: Output trajectory must NOT contain NaNs
    assert not jnp.any(jnp.isnan(results.states)), \
        "Simulation results contain NaNs! Safety invariant violated."

    # 2. Error Reporting: Controller error flag should be set
    assert "error" in results.controller_keys, "Controller error flag missing from results"
    error_idx = results.controller_keys.index("error")
    errors = results.controller_values[error_idx]

    assert jnp.any(errors), "Simulation did not report error despite NaN generation."

    # 3. Error Code: Specific code -10 for Integration NaN
    assert "error_data" in results.controller_keys, "Error data missing"
    error_data_idx = results.controller_keys.index("error_data")
    error_codes = results.controller_values[error_data_idx]

    has_nan_error = jnp.any(error_codes == INTEGRATION_NAN_ERROR)

    assert has_nan_error, \
        f"Expected error code {INTEGRATION_NAN_ERROR} (INTEGRATION_NAN_ERROR), but got {error_codes}"

    # 4. Trajectory behavior
    final_state = results.states[-1, 0]
    # State should be <= 0.6 (where it blows up) and definitely not NaN
    assert final_state <= 0.6, \
           f"Final state {final_state} exceeded expected bounds."

if __name__ == "__main__":
    test_integration_nan_handling(use_jit=False)
    test_integration_nan_handling(use_jit=True)
    print("Test passed!")
