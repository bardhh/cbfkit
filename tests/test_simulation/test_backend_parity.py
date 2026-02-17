import jax.numpy as jnp
import pytest

import cbfkit.simulation.simulator as sim
from cbfkit.integration import forward_euler, runge_kutta_4


def _dynamics(x):
    # 2D fully actuated integrator x_dot = u
    return jnp.zeros_like(x), jnp.eye(x.shape[0])


def _legacy_controller(_t, x):
    # Legacy 2-arg controller signature to exercise setup_controller adapter.
    return -0.1 * x


@pytest.mark.parametrize("integrator", [forward_euler, runge_kutta_4])
def test_python_and_jit_backend_parity(integrator):
    x0 = jnp.array([1.0, -0.5])

    results_py = sim.execute(
        x0=x0,
        dt=0.05,
        num_steps=20,
        dynamics=_dynamics,
        integrator=integrator,
        controller=_legacy_controller,  # type: ignore[arg-type]
        use_jit=False,
        verbose=False,
    )
    results_jit = sim.execute(
        x0=x0,
        dt=0.05,
        num_steps=20,
        dynamics=_dynamics,
        integrator=integrator,
        controller=_legacy_controller,  # type: ignore[arg-type]
        use_jit=True,
        verbose=False,
    )

    # Python path returns post-integration states for each step (x_{k+1}),
    # while JIT path logs the pre-integration state at each step (x_k).
    # Compare the overlapping horizon directly (x_1..x_{N-1}).
    assert jnp.allclose(results_py.states[:-1], results_jit.states[1:], rtol=1e-6, atol=1e-6)

    # For this integrator setup (x_dot=u, u held per step), the final Python state
    # should match one extra forward step from the last JIT state/control sample.
    dt = 0.05
    expected_last_state = results_jit.states[-1] + results_jit.controls[-1] * dt
    assert jnp.allclose(results_py.states[-1], expected_last_state, rtol=1e-6, atol=1e-6)

    assert jnp.allclose(results_py.controls, results_jit.controls, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(results_py.estimates, results_jit.estimates, rtol=1e-6, atol=1e-6)
