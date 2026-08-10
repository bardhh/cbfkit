"""Regression tests for eager-path memory, PRNG, and JIT-cache efficiency.

Three separate defects are pinned here:

1. The eager loop used to log a growing ``planner_data.xs`` slice every step,
   retaining O(N^2) memory for a field ``format_return_data`` then dropped.
2. Each eager step split the PRNG key four times instead of once.
3. ``execute()`` built a fresh progress-hook closure per call, and because the
   hook is a static argument of the jitted simulator, every verbose run
   recompiled.
"""

import warnings

import jax.numpy as jnp
import numpy as np
from jax import random

from cbfkit.estimators.naive import naive as estimator
from cbfkit.integration import forward_euler
from cbfkit.sensors import perfect as sensor
from cbfkit.simulation import simulator
from cbfkit.utils.jit_monitor import JitMonitor
from cbfkit.utils.user_types import PlannerData


def _dynamics(x):
    return jnp.zeros_like(x), jnp.eye(len(x))


def _planner(t, x, u_prev, key, data):
    """Minimal control-trajectory planner: enough to enable the trajectory buffer."""
    return jnp.zeros(2), data._replace(u_traj=jnp.zeros((2, 1)))


def _controller(t, x, u_nom, key, data):
    return u_nom, data


def _run_eager(num_steps, **kwargs):
    return simulator.execute(
        x0=jnp.array([-0.9, -0.9]),
        dt=0.01,
        num_steps=num_steps,
        dynamics=_dynamics,
        integrator=forward_euler,
        sensor=sensor,
        estimator=estimator,
        use_jit=False,
        verbose=False,
        **kwargs,
    )


def test_planner_xs_not_retained_or_dropped():
    """A planner-driven run logs no growing-shape planner field and warns about none.

    300 steps suffice: the old defect logged a distinct (growing) shape at every
    step and would retain ~720 KB here, an order of magnitude over the bound.
    """
    num_steps = 300
    steps = []
    real_cls = simulator.SimulationStepData

    class _Spy(real_cls):  # type: ignore[misc,valid-type]
        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            obj = real_cls.__new__(cls, *args, **kwargs)
            steps.append(obj)
            return obj

    simulator.SimulationStepData = _Spy
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _run_eager(num_steps, planner=_planner, controller=_controller)
    finally:
        simulator.SimulationStepData = real_cls

    assert len(steps) == num_steps

    # No logged planner field may change shape across timesteps: that is both the
    # O(N^2) retention and the reason format_return_data dropped the field.
    for idx, field in enumerate(PlannerData._fields):
        shapes = {
            np.shape(s.planner_values[idx])
            for s in steps
            if s.planner_values[idx] is not None and hasattr(s.planner_values[idx], "shape")
        }
        assert len(shapes) <= 1, (
            f"planner field {field!r} logged {len(shapes)} distinct shapes "
            f"(e.g. {sorted(shapes)[:3]} ... {sorted(shapes)[-1]})"
        )

    retained = sum(int(v.nbytes) for s in steps for v in s.planner_values if hasattr(v, "nbytes"))
    # The old behaviour retained ~num_steps^2 * 8 bytes (~720 KB at 300 steps,
    # one growing trajectory slice per step); everything legitimately logged is
    # fixed-size and far smaller.
    assert retained < 150_000, f"retained {retained / 1e6:.1f} MB of planner data"

    dropped = [str(c.message) for c in caught if "dropped from SimulationResults" in str(c.message)]
    assert dropped == [], f"fields dropped for inconsistent shapes: {dropped}"


def test_one_key_split_per_eager_step():
    """The eager step splits the PRNG key exactly once per timestep.

    The assertion is an exact count, so 100 steps discriminate as well as 800.
    """
    num_steps = 100
    real_split = random.split
    calls = {"n": 0}

    def counting_split(*args, **kwargs):
        calls["n"] += 1
        return real_split(*args, **kwargs)

    random.split = counting_split
    try:
        _run_eager(
            num_steps,
            planner=_planner,
            controller=_controller,
            nominal_controller=lambda t, x, key, ref: (jnp.zeros(2), None),
        )
    finally:
        random.split = real_split

    assert calls["n"] == num_steps, f"expected {num_steps} splits, got {calls['n']}"


def test_jit_progress_hook_reuses_compilation():
    """Two identical verbose progress runs share one compilation of simulator_jit."""
    JitMonitor.reset()

    def run():
        simulator.execute(
            x0=jnp.array([0.0, 0.0]),
            dt=0.01,
            num_steps=10,
            dynamics=_dynamics,
            integrator=forward_euler,
            use_jit=True,
            jit_progress=True,
            jit_progress_interval=5,
            verbose=True,
        )

    run()
    assert JitMonitor.get_counts().get("simulator_jit", 0) == 1

    run()
    assert (
        JitMonitor.get_counts().get("simulator_jit", 0) == 1
    ), "progress hook identity changed between calls, forcing a recompile"
