"""Single integrator sweep benchmark.

Registers the tutorial_single_integrator_sweep scenario for CLI use.
See tutorials/parameter_sweep_tutorial.py for the full walkthrough.
"""

from __future__ import annotations

from cbfkit.benchmarks.registry import register_sweepable_scenario
from cbfkit.benchmarks.scenario_builders import (
    build_single_integrator_setup,
    compute_sweep_metrics,
    resolve_circular_obstacles,
)
from cbfkit.simulation.monte_carlo_gpu import conduct_monte_carlo_gpu
from cbfkit.simulation.safety_verification import compute_safety_statistics

N_TRIALS = 1


@register_sweepable_scenario(
    "tutorial_single_integrator_sweep",
    sweepable_params=["alpha", "control_limit", "n_obstacles"],
    description="Single integrator obstacle avoidance with sweepable CBF parameters.",
)
def tutorial_sweep(seed: int, params: dict) -> dict:
    alpha = params.get("alpha", 1.0)
    control_limit = params.get("control_limit", 5.0)
    n_obstacles = int(params.get("n_obstacles", 2))
    obs = resolve_circular_obstacles(params)

    setup = build_single_integrator_setup(
        seed,
        n_obstacles=n_obstacles,
        alpha=alpha,
        control_limit=control_limit,
        obstacles=obs,
        dt=0.1,
        num_steps=100,
    )

    results = conduct_monte_carlo_gpu(setup, n_trials=N_TRIALS, seed=seed)
    stats = compute_safety_statistics(results)
    return compute_sweep_metrics(results, stats, setup)
