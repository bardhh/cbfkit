import jax.numpy as jnp
from jax import random
import pytest
from cbfkit.simulation import simulator as sim
from cbfkit.utils.user_types import ControllerData
from cbfkit.integration import forward_euler

def test_jit_structure_stability_and_optimization():
    """
    Regression test for JIT simulation structure stability and memory optimization.

    This test verifies two critical behaviors:
    1. Structure Stability (Priming):
       The `simulator.execute` function must "prime" the JIT loop by running the controller
       once before `lax.scan` begins. This ensures that if the controller returns
       dynamic structures (like `sub_data` with specific keys) that differ from the
       empty `initial_controller_data`, `lax.scan` receives a consistent carry structure
       and does not crash with a structure mismatch error.

    2. Memory Optimization (Stripping):
       `simulator_jit` includes logic to strip "solver_params" from the logged output
       (to save memory) but preserve other keys. This test asserts that "solver_params"
       is indeed absent from the results, while "custom_data" is preserved.
    """

    # 1. Setup simple dynamics (Scalar integrator: x_dot = u)
    def dynamics(x):
        return jnp.zeros_like(x), jnp.eye(x.shape[0])

    # 2. Define Mock Controller with specific sub_data structure
    def mock_controller(t, x, u_nom, key, data):
        # Return a sub_data with both a key to be stripped and a key to be kept
        sub_data = {
            "solver_params": jnp.ones((1,)),  # Should be STRIPPED from logs
            "custom_data": jnp.ones((1,))     # Should be KEPT in logs
        }

        return u_nom, ControllerData(sub_data=sub_data)

    # 3. Execute Simulation with JIT
    # Note: initial_controller_data passed internally will be empty/None.
    # The controller returns populated sub_data.
    # Without "priming", lax.scan would fail here.

    x0 = jnp.array([0.0])
    dt = 0.1
    num_steps = 5

    results = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=num_steps,
        dynamics=dynamics,
        integrator=forward_euler,
        controller=mock_controller,
        use_jit=True,
        verbose=False,
        key=random.PRNGKey(0)
    )

    # 4. Verifications

    # Check 1: Simulation completed (implicit by reaching here without crash)
    assert results.states.shape == (num_steps, 1)

    # Check 2: 'custom_data' is present (Structure stability + Data preservation)
    # The logging logic flattens dicts: sub_data["custom_data"] -> "sub_data_custom_data"
    # Wait, simulator.py flatten logic:
    # "controller_data_keys.append(f"{key_str}_{sub_k}")"
    # key_str for sub_data field is "sub_data"
    expected_key = "sub_data_custom_data"
    assert expected_key in results.controller_keys, \
        f"Expected '{expected_key}' in results. Found: {results.controller_keys}"

    # Check 3: 'solver_params' is ABSENT (Memory optimization)
    stripped_key = "sub_data_solver_params"
    assert stripped_key not in results.controller_keys, \
        f"Expected '{stripped_key}' to be stripped from results. Found: {results.controller_keys}"

    # Verify values are correct
    idx = results.controller_keys.index(expected_key)
    # mock_controller returns ones
    assert jnp.all(results.controller_values[idx] == 1.0)

if __name__ == "__main__":
    test_jit_structure_stability_and_optimization()
