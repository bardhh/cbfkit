
import jax.numpy as jnp
import pytest
from cbfkit.simulation import simulator
from cbfkit.simulation.simulator import SOLVER_STATUS_MAP

def nan_dynamics(x):
    # Returns NaNs immediately
    return jnp.full_like(x, jnp.nan), jnp.zeros((x.shape[0], 1))

def integrator(x, f, dt):
    return x + dt * f(x)

def test_jit_uncontrolled_nan_reporting():
    """
    Verifies that running an uncontrolled simulation (controller=None)
    with JIT enabled correctly reports INTEGRATION_NAN_ERROR (-10)
    when NaNs occur, instead of failing silently or with missing status.
    """
    x0 = jnp.array([1.0, 2.0])
    dt = 0.01
    num_steps = 10

    # Run with JIT, no controller
    results = simulator.execute(
        x0=x0,
        dt=dt,
        num_steps=num_steps,
        dynamics=nan_dynamics,
        integrator=integrator,
        use_jit=True,
        verbose=False
    )

    # Check that error_data is present
    assert "error_data" in results.controller_data, "error_data missing from results"

    err_data = results.controller_data["error_data"]

    # Check that we caught the specific error code (-10)
    # The simulation should stop at step 0, filling the rest with -10 or previous value
    # Since it fails immediately, we expect -10.
    assert jnp.any(err_data == -10), f"Expected status -10, got {err_data}"

    # Check that error flag is set
    assert "error" in results.controller_data
    assert jnp.any(results.controller_data["error"])

if __name__ == "__main__":
    test_jit_uncontrolled_nan_reporting()
