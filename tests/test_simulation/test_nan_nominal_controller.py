
import jax.numpy as jnp
import pytest
from cbfkit.simulation import simulator
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)

def dynamics(x):
    f = jnp.zeros_like(x)
    g = jnp.eye(2)
    return f, g

def nan_nominal_controller(t, x, key, reference):
    u_nom = jnp.array([jnp.nan, jnp.nan])
    return u_nom, {}

def h(t, x):
    return jnp.array(x[0] + 1.0)

def grad_h(t, x):
    return jnp.array([1.0, 0.0])

barrier = (
    [h],
    [grad_h],
    [lambda t, x: jnp.zeros((2,2))],
    [lambda t, x: jnp.zeros(())],
    [lambda x: 0.1]
)

setup_controller = cbf_clf_qp_generator(
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints
)

controller = setup_controller(
    control_limits=jnp.array([10.0, 10.0]),
    dynamics_func=dynamics,
    barriers=barrier,
    relaxable_cbf=True,
    slack_penalty_cbf=1.0,
)

def test_nan_nominal_controller_message(capsys):
    """Test that NaNs in nominal controller are explicitly reported."""
    x0 = jnp.array([0.0, 0.0])
    dt = 0.01
    num_steps = 5

    # Run simulation (JIT mode where jax.debug.print works)
    # Note: capturing jax.debug.print requires running without capturing or using specific flags/tools.
    # However, jax.debug.print usually writes to stdout/stderr which capsys captures.
    # But JIT compilation output might be tricky.

    # We use JIT mode as cbf_clf_qp_generator uses jax.debug.print inside JIT
    try:
        simulator.execute(
            x0=x0,
            dt=dt,
            num_steps=num_steps,
            dynamics=dynamics,
            integrator=lambda x, f, dt: x + f(x)*dt,
            nominal_controller=nan_nominal_controller,
            controller=controller,
            verbose=True,
            use_jit=True
        )
    except Exception:
        pass

    captured = capsys.readouterr()
    # Check for the specific message parts
    # Note: jax.debug.print output format depends on backend but usually shows up in stdout/stderr
    # If using CPU, it should be captured.

    # We look for "Sources: u_nom=True"
    # Or at least "u_nom="

    # Also "NAN_INPUT_DETECTED"

    # For robustness, we check if either stdout or stderr contains it.
    output = captured.out + captured.err

    # If the test environment doesn't capture JAX debug prints (common in some setups),
    # this assertion might fail. But let's try.
    # In the bash session earlier, I saw the output on stdout.

    assert "NAN_INPUT_DETECTED" in output
    assert "u_nom=" in output or "u_nom=True" in output
