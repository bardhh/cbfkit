import pytest
import jax.numpy as jnp
from cbfkit.simulation import simulator
from cbfkit.integration.forward_euler import forward_euler
from cbfkit.utils.user_types import ControllerData
# We import INTEGRATION_NAN_ERROR from simulator_jit as it is defined there
from cbfkit.simulation.simulator_jit import INTEGRATION_NAN_ERROR

def test_integration_nan_handling():
    """
    Test that the simulator gracefully handles NaNs generated during integration.
    It should stop the simulation, set the error flag, record INTEGRATION_NAN_ERROR (-10),
    and clamp the state to the last valid value (preventing NaNs in trajectory).
    """
    # 1. Define unstable dynamics that produce NaN when state exceeds 1.0
    def unstable_dynamics(x):
        # x_dot = NaN if x[0] > 1.0 else 10.0
        # With x0=0, dt=0.01, it reaches > 1.0 around step 10.
        f = jnp.where(x[0] > 1.0, jnp.nan * jnp.ones(1), 10.0 * jnp.ones(1))
        g = jnp.zeros((1, 1))
        return f, g

    # 2. Setup
    x0 = jnp.array([0.0])
    dt = 0.01
    tf = 0.5 # 50 steps
    num_steps = int(tf / dt)

    # Note: For JIT mode, we must initialize error_data to an array if we want it to be tracked
    # and updated from None. If it starts as None, JAX keeps it as None (static).
    # Ideally simulator.execute should handle this, but currently we may need to supply it.
    # Let's see if we can trigger the fix in simulator.py later. For now, we manually init.
    # actually, I'll let the test fail if execute doesn't handle it, to prove the need for fix.
    # But to test the JIT logic itself (in simulator_jit), I should provide it.

    # We will initialize controller_data with a dummy error_data for JIT to track it.
    init_c_data = ControllerData(error_data=jnp.array(0))

    # Dummy nominal controller to satisfy simulator requirements
    def nominal_controller(t, x, key, ref):
        return jnp.zeros((1,)), ControllerData()

    # 3. Test JIT Mode
    results_jit = simulator.execute(
        x0=x0, dt=dt, num_steps=num_steps,
        dynamics=unstable_dynamics,
        integrator=forward_euler,
        nominal_controller=nominal_controller,
        use_jit=True,
        verbose=False,
        controller_data=init_c_data
    )

    # Verify JIT error handling
    assert jnp.any(results_jit.controller_data["error"]), "JIT: Simulation should stop with error."

    # Find the first error index
    error_idx_jit = jnp.argmax(results_jit.controller_data["error"])
    status_code_jit = results_jit.controller_data["error_data"][error_idx_jit]

    assert status_code_jit == INTEGRATION_NAN_ERROR, \
        f"JIT: Expected error code {INTEGRATION_NAN_ERROR}, got {status_code_jit}"

    # Verify JIT state clamping (no NaNs)
    assert not jnp.any(jnp.isnan(results_jit.states)), \
        "JIT: Trajectory should not contain NaNs (should be clamped)."


    # 4. Test Python Mode (Expected to Fail initially due to -1 code and NaNs in trajectory)
    results_py = simulator.execute(
        x0=x0, dt=dt, num_steps=num_steps,
        dynamics=unstable_dynamics,
        integrator=forward_euler,
        nominal_controller=nominal_controller,
        use_jit=False,
        verbose=False,
        controller_data=init_c_data
    )

    # Verify Python error handling
    assert jnp.any(results_py.controller_data["error"]), "Python: Simulation should stop with error."

    error_idx_py = jnp.argmax(results_py.controller_data["error"])
    status_code_py = results_py.controller_data["error_data"][error_idx_py]

    assert status_code_py == INTEGRATION_NAN_ERROR, \
        f"Python: Expected error code {INTEGRATION_NAN_ERROR}, got {status_code_py}"

    # Verify Python state clamping (no NaNs)
    # The current Python implementation appends the NaN state before breaking, so this will fail.
    assert not jnp.any(jnp.isnan(results_py.states)), \
        "Python: Trajectory should not contain NaNs (should be clamped)."

if __name__ == "__main__":
    test_integration_nan_handling()
