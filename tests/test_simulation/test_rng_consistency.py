
import jax.numpy as jnp
from jax import random
from cbfkit.simulation.simulator import execute
from cbfkit.utils.user_types import ControllerData, PlannerData
import pytest

def dynamics(x):
    return jnp.zeros_like(x), jnp.eye(len(x))

def integrator(x, f, dt):
    return x

def test_rng_consistency_between_backends():
    """
    Verifies that the Python backend and JIT backend produce identical
    sequences of random keys for Planners and Controllers given the same seed.
    This also ensures that Planner and Controller receive different keys (no correlation).
    """
    x0 = jnp.zeros(2)
    dt = 0.1
    num_steps = 1

    # Define components that capture the key
    # We use the data return path to extract the key value from the opaque simulation loop

    def planner(t, x, u_prev, key, data):
        # Store key[1] in prev_robustness (must be Array for JIT compatibility)
        return jnp.zeros(2), data._replace(prev_robustness=jnp.array(key[1], dtype=float))

    def controller(t, x, u_nom, key, data):
        # Store key[1] in u (must be Array for JIT compatibility)
        return jnp.zeros(2), data._replace(u=jnp.array([key[1], 0.0], dtype=float))

    # Initialize data structures for JIT structure matching
    init_c_data = ControllerData(u=jnp.zeros(2))
    init_p_data = PlannerData(prev_robustness=jnp.array(0.0, dtype=float))

    # 1. Run JIT Backend (The "Golden" Reference)
    results_jit = execute(
        x0=x0, dt=dt, num_steps=num_steps,
        dynamics=dynamics, integrator=integrator,
        planner=planner, controller=controller,
        controller_data=init_c_data,
        planner_data=init_p_data,
        use_jit=True, verbose=False
    )

    p_key_jit = float(results_jit.planner_data['prev_robustness'][0])
    c_key_jit = float(results_jit.controller_data['u'][0][0])

    # Assert independence
    assert p_key_jit != c_key_jit, "JIT: Planner and Controller received the same key!"

    # 2. Run Python Backend
    results_py = execute(
        x0=x0, dt=dt, num_steps=num_steps,
        dynamics=dynamics, integrator=integrator,
        planner=planner, controller=controller,
        controller_data=init_c_data, # Pass init data for consistency, though optional for Python
        planner_data=init_p_data,
        use_jit=False, verbose=False
    )

    p_key_py = float(results_py.planner_data['prev_robustness'][0])
    c_key_py = float(results_py.controller_data['u'][0][0])

    # Assert independence
    assert p_key_py != c_key_py, "Python: Planner and Controller received the same key!"

    # Assert Consistency
    assert p_key_py == p_key_jit, f"Planner keys differ! Py: {p_key_py}, JIT: {p_key_jit}"
    assert c_key_py == c_key_jit, f"Controller keys differ! Py: {c_key_py}, JIT: {c_key_jit}"
