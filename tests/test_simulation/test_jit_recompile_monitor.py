"""Regression tests for simulator JIT recompilation behavior."""

import jax.numpy as jnp

from cbfkit.integration import forward_euler
from cbfkit.simulation import simulator
from cbfkit.utils.jit_monitor import JitMonitor


def _run_simulation(dynamics_func):
    simulator.execute(
        x0=jnp.array([0.0, 0.0]),
        dt=0.01,
        num_steps=10,
        dynamics=dynamics_func,
        integrator=forward_euler,
        use_jit=True,
        verbose=False,
    )


def test_simulator_jit_recompile_behavior() -> None:
    """Compile once for a static call shape and recompile for new static callables."""
    JitMonitor.reset()

    def dynamics_v1(x):
        return jnp.zeros_like(x), jnp.zeros((2, 1))

    _run_simulation(dynamics_v1)
    compile_count_1 = JitMonitor.get_counts().get("simulator_jit", 0)
    assert compile_count_1 == 1

    _run_simulation(dynamics_v1)
    compile_count_2 = JitMonitor.get_counts().get("simulator_jit", 0)
    assert compile_count_2 == 1

    def dynamics_v2(x):
        return jnp.zeros_like(x), jnp.zeros((2, 1))

    _run_simulation(dynamics_v2)
    compile_count_3 = JitMonitor.get_counts().get("simulator_jit", 0)
    assert compile_count_3 == 2
