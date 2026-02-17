
import os
import pytest
import jax.numpy as jnp
from jax import random
from cbfkit.simulation import simulator

def simple_dynamics(x):
    return jnp.zeros_like(x), jnp.eye(len(x))

def simple_integrator(x, f, dt):
    return x

def key_logger_controller(t, x, u_nom, key, data):
    # Log key as control input to verify it changes
    u = jnp.array([key[0].astype(float)])
    return u, data

@pytest.fixture
def clean_env():
    old_seed = os.environ.get("CBFKIT_SEED")
    if "CBFKIT_SEED" in os.environ:
        del os.environ["CBFKIT_SEED"]
    yield
    if old_seed is not None:
        os.environ["CBFKIT_SEED"] = old_seed
    else:
        if "CBFKIT_SEED" in os.environ:
            del os.environ["CBFKIT_SEED"]

def run_sim(seed_env=None, key=None):
    if seed_env is not None:
        os.environ["CBFKIT_SEED"] = str(seed_env)

    res = simulator.execute(
        x0=jnp.array([0.0]),
        dt=0.1,
        num_steps=1,
        dynamics=simple_dynamics,
        integrator=simple_integrator,
        controller=key_logger_controller,
        key=key,
        verbose=False
    )
    return res.controls[0, 0]

def test_cbfkit_seed_impact(clean_env):
    """Test that CBFKIT_SEED changes the simulation key."""
    # Ensure no env var for default run
    if "CBFKIT_SEED" in os.environ:
        del os.environ["CBFKIT_SEED"]
    val_default = run_sim(seed_env=None)

    val_seeded = run_sim(seed_env=12345)

    # 1. Impact: Seed changes result
    assert val_default != val_seeded

    # 2. Determinism: Same seed -> Same result
    val_seeded_2 = run_sim(seed_env=12345)
    assert val_seeded == val_seeded_2

def test_explicit_key_precedence(clean_env):
    """Test that explicit key overrides CBFKIT_SEED."""
    # Env seed 12345
    val_env_only = run_sim(seed_env=12345)

    # Explicit key (seed 999)
    key_explicit = random.PRNGKey(999)
    # We pass seed_env=12345 to set the env var, but pass key to override it
    val_explicit = run_sim(seed_env=12345, key=key_explicit)

    # 1. Precedence: Explicit key changes result vs Env only
    assert val_explicit != val_env_only

    # 2. Determinism: Explicit key gives consistent result regardless of env
    if "CBFKIT_SEED" in os.environ:
        del os.environ["CBFKIT_SEED"]
    val_explicit_2 = run_sim(seed_env=None, key=key_explicit)
    assert val_explicit == val_explicit_2

def test_invalid_seed_fallback(clean_env):
    """Test fallback to 0 if CBFKIT_SEED is invalid."""
    os.environ["CBFKIT_SEED"] = "not_an_integer"
    # Calling run_sim without args uses existing env
    res = simulator.execute(
        x0=jnp.array([0.0]),
        dt=0.1,
        num_steps=1,
        dynamics=simple_dynamics,
        integrator=simple_integrator,
        controller=key_logger_controller,
        verbose=False
    )
    val_invalid = res.controls[0, 0]

    # Should match default seed 0
    del os.environ["CBFKIT_SEED"]
    val_default = run_sim(seed_env=None)

    assert val_invalid == val_default
