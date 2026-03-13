"""GPU-accelerated Monte Carlo speedup benchmark scenario."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

from cbfkit.benchmarks.registry import register_scenario
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_vanilla_clf_constraints,
    generate_compute_zeroing_cbf_constraints,
)
from cbfkit.integration import forward_euler
from cbfkit.simulation.monte_carlo_gpu import (
    MonteCarloSetup,
    conduct_monte_carlo_gpu,
)
from cbfkit.simulation.safety_verification import compute_safety_statistics
from cbfkit.utils.user_types import CertificateCollection, ControllerData, PlannerData


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dynamics(x):
    return jnp.zeros(2), jnp.eye(2)


def _make_cbf(c, r):
    return (
        lambda _t, x: jnp.sum((x - c) ** 2) - r**2,
        lambda _t, x: 2 * (x - c),
        lambda _t, _x: 2 * jnp.eye(2),
        lambda _t, _x: 0.0,
        lambda h: 1.0 * h,
    )


def _default_sensor(t, x, *, sigma=None, key=None):
    return x


def _default_estimator(t, y, z, u, c):
    return y, c if c is not None else jnp.zeros((len(y), len(y)))


def _default_perturbation(x, u, f, g):
    def p(key):
        return jnp.zeros_like(x)
    return p


def _build_setup(seed: int, n_obstacles: int = 5):
    """Build a ``MonteCarloSetup`` with random circular obstacles."""
    key = random.PRNGKey(seed)
    key_c, key_r = random.split(key)
    centers = random.uniform(key_c, (n_obstacles, 2), minval=2.0, maxval=8.0)
    radii = random.uniform(key_r, (n_obstacles,), minval=0.5, maxval=1.0)

    barrier_tuples = [_make_cbf(centers[i], radii[i]) for i in range(n_obstacles)]
    barriers = CertificateCollection(*[list(x) for x in zip(*barrier_tuples)])

    controller = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints,
    )(
        control_limits=jnp.array([5.0, 5.0]),
        dynamics_func=_dynamics,
        barriers=barriers,
        relaxable_cbf=False,
        relaxable_clf=True,
    )

    def nominal_controller(t, x, _k, _r):
        return 2.0 * (jnp.array([10.0, 10.0]) - x), None

    def initial_state_sampler(key):
        return random.uniform(key, (2,), minval=-1.0, maxval=1.0)

    # Prime controller data by calling controller once
    prime_key = random.PRNGKey(42)
    x0_probe = jnp.zeros(2)
    f_d, g_d = _dynamics(x0_probe)
    u_nom_d = jnp.zeros((g_d.shape[1],))
    _, c_data = controller(0.0, x0_probe, u_nom_d, prime_key, ControllerData())

    return MonteCarloSetup(
        dt=0.01,
        num_steps=200,
        dynamics=_dynamics,
        integrator=forward_euler,
        initial_state_sampler=initial_state_sampler,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=_default_sensor,
        estimator=_default_estimator,
        perturbation=_default_perturbation,
        sigma=jnp.zeros(0),
        controller_data=c_data,
        planner=None,
        planner_data=PlannerData(),
    )


# ---------------------------------------------------------------------------
# Benchmark scenario
# ---------------------------------------------------------------------------

TRIAL_COUNTS = [100, 500, 1000]


@register_scenario(
    "monte_carlo_gpu_speedup",
    description="Monte Carlo speedup benchmark: measures throughput at varying trial counts.",
)
def monte_carlo_gpu_speedup(seed: int) -> dict[str, float | int]:
    """Run GPU Monte Carlo at several trial counts and report timing + safety."""
    setup = _build_setup(seed)
    platform = jax.devices()[0].platform

    metrics: dict[str, float | int] = {"platform": platform}

    for n in tqdm(TRIAL_COUNTS, desc=f"  Trials (seed={seed})", unit="batch", leave=False):
        results = conduct_monte_carlo_gpu(setup, n_trials=n, seed=seed)
        stats = compute_safety_statistics(results)

        metrics[f"time_{n}"] = results.wall_time_s
        metrics[f"trials_per_sec_{n}"] = n / results.wall_time_s
        metrics[f"violation_rate_{n}"] = stats.violation_rate
        metrics[f"min_barrier_{n}"] = stats.min_barrier_value

    # Summary fields expected by the metrics module
    metrics["success"] = 1
    metrics["safety_violations"] = int(any(
        metrics.get(f"violation_rate_{n}", 0) > 0 for n in TRIAL_COUNTS
    ))
    metrics["solver_failures"] = 0
    metrics["avg_step_ms"] = metrics[f"time_{TRIAL_COUNTS[0]}"] / setup.num_steps * 1000.0

    return metrics
