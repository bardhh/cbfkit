"""GPU-accelerated Monte Carlo speedup benchmark scenario."""

from __future__ import annotations

import jax

from cbfkit.benchmarks._progress import make_progress
from cbfkit.benchmarks.registry import register_scenario, register_sweepable_scenario
from cbfkit.benchmarks.scenario_builders import (
    build_single_integrator_setup,
    compute_sweep_metrics,
    resolve_circular_obstacles,
)
from cbfkit.simulation.monte_carlo_gpu import conduct_monte_carlo_gpu
from cbfkit.simulation.safety_verification import compute_safety_statistics


# ---------------------------------------------------------------------------
# Original benchmark scenario (non-sweep)
# ---------------------------------------------------------------------------

TRIAL_COUNTS = [100, 500, 1000]


@register_scenario(
    "monte_carlo_gpu_speedup",
    description="Monte Carlo speedup benchmark: measures throughput at varying trial counts.",
)
def monte_carlo_gpu_speedup(seed: int) -> dict[str, float | int]:
    """Run GPU Monte Carlo at several trial counts and report timing + safety."""
    setup = build_single_integrator_setup(seed)
    platform = jax.devices()[0].platform

    metrics: dict[str, float | int] = {"platform": platform}

    progress = make_progress(transient=True)
    with progress:
        task = progress.add_task(f"Trials (seed={seed})", total=len(TRIAL_COUNTS))
        for n in TRIAL_COUNTS:
            results = conduct_monte_carlo_gpu(setup, n_trials=n, seed=seed)
            stats = compute_safety_statistics(results)

            metrics[f"time_{n}"] = results.wall_time_s
            metrics[f"trials_per_sec_{n}"] = n / results.wall_time_s
            metrics[f"violation_rate_{n}"] = stats.violation_rate
            metrics[f"min_barrier_{n}"] = stats.min_barrier_value
            progress.advance(task)

    # Summary fields expected by the metrics module
    metrics["success"] = 1
    metrics["safety_violations"] = int(any(
        metrics.get(f"violation_rate_{n}", 0) > 0 for n in TRIAL_COUNTS
    ))
    metrics["solver_failures"] = 0
    metrics["avg_step_ms"] = metrics[f"time_{TRIAL_COUNTS[0]}"] / setup.num_steps * 1000.0

    return metrics


# ---------------------------------------------------------------------------
# Sweepable benchmark scenario
# ---------------------------------------------------------------------------

SWEEP_TRIAL_COUNT = 100


@register_sweepable_scenario(
    "monte_carlo_gpu_sweep",
    sweepable_params=["alpha", "control_limit", "n_obstacles", "relaxable_cbf"],
    description="GPU Monte Carlo with sweepable CBF parameters.",
)
def monte_carlo_gpu_sweep(seed: int, params: dict) -> dict[str, float | int]:
    """Sweepable version: accepts CBF parameters and runs a fixed trial count."""
    alpha = params.get("alpha", 1.0)
    control_limit = params.get("control_limit", 5.0)
    n_obstacles = int(params.get("n_obstacles", 5))
    relaxable = bool(params.get("relaxable_cbf", False))
    obs = resolve_circular_obstacles(params)

    setup = build_single_integrator_setup(
        seed,
        n_obstacles=n_obstacles,
        alpha=alpha,
        control_limit=control_limit,
        relaxable_cbf=relaxable,
        obstacles=obs,
    )

    results = conduct_monte_carlo_gpu(setup, n_trials=SWEEP_TRIAL_COUNT, seed=seed)
    stats = compute_safety_statistics(results)
    metrics = compute_sweep_metrics(results, stats, setup)
    metrics["platform"] = jax.devices()[0].platform
    metrics["n_trials"] = SWEEP_TRIAL_COUNT
    metrics["trials_per_sec"] = SWEEP_TRIAL_COUNT / results.wall_time_s
    return metrics
