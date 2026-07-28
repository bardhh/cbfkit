"""Benchmark: render one CBF scene through every ``CBFAnimator`` backend.

Runs the README quick-start CBF simulation (unicycle reach-avoid: the robot
drives to a goal while a CBF safety filter keeps it clear of an obstacle) once,
then renders the *same* declarative scene through each available backend and
reports wall-clock render time and output-file size.  This is the reproducible
source of the backend-comparison table in the Manim-2D-backend PR.

The plotly row is skipped if plotly is not installed; the ``manim-*`` rows are
skipped (with a note) if the ``manim`` extra is absent.  Set ``CBFKIT_TEST_MODE``
to shrink the simulation for a quick smoke run.

Requires the ``manim`` extra (``pip install cbfkit[manim]``) plus ffmpeg, and on
macOS the cairo/pango libraries (``brew install ffmpeg cairo pango``) for the
manim rows.

Run:

    python benchmarks/animator_backend_comparison.py
"""

import os
import platform
import sys
import tempfile
import time

import jax.numpy as jnp
import numpy as np
from jax import jit

from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.estimators import naive
from cbfkit.integration import runge_kutta_4
from cbfkit.sensors import perfect
from cbfkit.simulation import simulator
from cbfkit.systems.unicycle.models.olfatisaber2002approximate.dynamics import (
    approx_unicycle_dynamics,
)
from cbfkit.utils.animator import CBFAnimator
from cbfkit.utils.animators.deps import _HAS_MANIM

# Scene constants (identical to examples/unicycle/reach_goal/manim_2d_animation.py).
GOAL = jnp.array([4.0, 0.0])
OBSTACLE_CENTER = jnp.array([2.0, 0.5, 0.0])
OBSTACLE_RADII = jnp.array([0.5, 0.5])
DT = 1e-2


def run_simulation():
    """Run the quick-start CBF sim and return ``(states, num_frames)``."""
    initial_state = jnp.array([0.0, 0.0, 0.0])
    actuation_limits = jnp.array([5.0, jnp.pi])
    num_steps = 500 if not os.getenv("CBFKIT_TEST_MODE") else 50

    dynamics = approx_unicycle_dynamics(lam=1.0)  # state: [x, y, theta]

    @jit
    def nominal_controller(t, state, key, data):
        x, y, th = state
        heading = jnp.arctan2(GOAL[1] - y, GOAL[0] - x)
        return (
            jnp.array(
                [
                    jnp.linalg.norm(jnp.array([x - GOAL[0], y - GOAL[1]])),  # speed
                    jnp.arctan2(jnp.sin(heading - th), jnp.cos(heading - th)),  # steering
                ]
            ),
            {},
        )

    cbf_factory, _, _ = ellipsoidal_barrier_factory(
        system_position_indices=(0, 1),
        obstacle_position_indices=(0, 1),
        ellipsoid_axis_indices=(0, 1),
    )
    barrier = rectify_relative_degree(
        function=cbf_factory(OBSTACLE_CENTER, OBSTACLE_RADII),
        system_dynamics=dynamics,
        state_dim=3,
        form="exponential",
    )(certificate_conditions=linear_class_k(10.0))

    controller = vanilla_cbf_clf_qp_controller(
        control_limits=actuation_limits,
        dynamics_func=dynamics,
        barriers=concatenate_certificates(barrier),
    )

    results = simulator.execute(
        x0=initial_state,
        dt=DT,
        num_steps=num_steps,
        dynamics=dynamics,
        integrator=runge_kutta_4,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=perfect,
        estimator=naive,
    )
    states = np.asarray(results.states)
    return states, states.shape[0]


def build_animator(states, backend):
    """Construct a CBFAnimator on ``states`` with the shared scene applied."""
    anim = CBFAnimator(
        states,
        dt=DT,
        backend=backend,
        title="CBF Safety Filter: Unicycle Reach-Avoid",
        aspect="equal",
        x_lim=(-0.7, 4.7),
        y_lim=(-1.2, 1.8),
    )
    anim.add_goal(np.asarray(GOAL), radius=0.25, color="g", label="Goal")
    anim.add_obstacle(np.asarray(OBSTACLE_CENTER[:2]), radius=float(OBSTACLE_RADII[0]), alpha=0.3)
    anim.add_agent(x_idx=0, y_idx=1, body_radius=0.12, body_color="tab:blue", trail=True)
    anim.show_time()
    return anim


def _fmt_size(num_bytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if num_bytes < 1024 or unit == "GiB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def time_backend(states, backend):
    """Render ``states`` through ``backend``; return ``(seconds, bytes, fmt)``."""
    ext = "html" if backend == "plotly" else "mp4"
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, f"bench_out.{ext}")
        anim = build_animator(states, backend)
        start = time.time()
        saved = anim.save(out_path)
        elapsed = time.time() - start
        size = os.path.getsize(saved)
    return elapsed, size, ext.upper()


def _plotly_available():
    try:
        import plotly  # noqa: F401
    except ImportError:
        return False
    return True


def main():
    print("Running CBF simulation (unicycle reach-avoid)...")
    states, num_frames = run_simulation()
    print(f"Simulated {num_frames} frames.\n")

    backends = []
    if _plotly_available():
        backends.append("plotly")
    else:
        print("plotly not installed — skipping the plotly row.")
    backends.append("matplotlib")
    if _HAS_MANIM:
        backends += ["manim-low", "manim-medium", "manim-high"]
    else:
        print("manim extra not installed — skipping the manim-* rows.")

    if os.getenv("CBFKIT_TEST_MODE"):
        print("CBFKIT_TEST_MODE set: timing matplotlib only as a smoke check.")
        backends = [b for b in backends if b == "matplotlib"]

    header = f"{'Backend':<14} {'Render time':>12} {'Output size':>12}  {'Format':<6}"
    print("\n" + header)
    print("-" * len(header))
    rows = []
    for backend in backends:
        elapsed, size, fmt = time_backend(states, backend)
        rows.append((backend, elapsed, size, fmt))
        print(f"{backend:<14} {elapsed:>10.1f} s {_fmt_size(size):>12}  {fmt:<6}")

    print(
        f"\nEnvironment: Python {platform.python_version()}, "
        f"manim {'installed' if _HAS_MANIM else 'absent'}, {num_frames} frames."
    )
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
