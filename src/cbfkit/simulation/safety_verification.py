"""Aggregate safety statistics from GPU Monte Carlo results."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from cbfkit.simulation.monte_carlo_gpu import MonteCarloGPUResults


class SafetyStatistics(NamedTuple):
    """Safety metrics aggregated across Monte Carlo trajectories."""

    n_trials: int
    violation_rate: float  # fraction of trials with any h(x) < 0
    min_barrier_value: float  # global min h(x) across all trials/steps
    mean_min_barrier: float  # mean of per-trial min h(x)
    per_trial_min_barrier: jnp.ndarray  # (n_trials,)
    per_trial_violated: jnp.ndarray  # (n_trials,) bool
    total_violation_steps: int
    solver_failure_rate: float


def compute_safety_statistics(
    results: MonteCarloGPUResults,
    barrier_key: str = "bfs",
    solver_status_key: str = "solver_status",
    violation_threshold: float = -1e-4,
) -> SafetyStatistics:
    """Compute safety metrics from batched Monte Carlo results.

    Extracts barrier function values and solver status from the batched
    controller data and computes aggregate statistics.

    Args:
        results: Output from ``conduct_monte_carlo_gpu``.
        barrier_key: Key for barrier values in ``controller_datas.sub_data``.
        solver_status_key: Key for solver status in ``controller_datas.sub_data``.
        violation_threshold: Barrier value below which a violation is counted.

    Returns:
        ``SafetyStatistics`` with per-trial and aggregate metrics.
    """
    c_datas = results.controller_datas
    n_trials = results.n_trials

    # Extract barrier function values from sub_data
    # Shape: (n_trials, num_steps, n_barriers)
    bfs = None
    if c_datas.sub_data is not None:
        sub = c_datas.sub_data
        if isinstance(sub, dict):
            bfs = sub.get(barrier_key)

    if bfs is None:
        # No barrier data available — return zero-violation statistics
        zeros = jnp.zeros(n_trials)
        return SafetyStatistics(
            n_trials=n_trials,
            violation_rate=0.0,
            min_barrier_value=0.0,
            mean_min_barrier=0.0,
            per_trial_min_barrier=zeros,
            per_trial_violated=jnp.zeros(n_trials, dtype=bool),
            total_violation_steps=0,
            solver_failure_rate=0.0,
        )

    # Per-step violation: any barrier below threshold
    # bfs shape: (n_trials, num_steps, n_barriers) or (n_trials, num_steps)
    if bfs.ndim == 3:
        step_violated = jnp.any(bfs < violation_threshold, axis=-1)  # (n_trials, num_steps)
        per_trial_min = jnp.min(bfs, axis=(1, 2))  # (n_trials,)
    else:
        step_violated = bfs < violation_threshold  # (n_trials, num_steps)
        per_trial_min = jnp.min(bfs, axis=1)  # (n_trials,)

    per_trial_violated = jnp.any(step_violated, axis=1)  # (n_trials,)
    total_violation_steps = int(jnp.sum(step_violated))
    violation_rate = float(jnp.mean(per_trial_violated))
    min_barrier_value = float(jnp.min(bfs))
    mean_min_barrier = float(jnp.mean(per_trial_min))

    # Solver failure rate
    solver_failure_rate = 0.0
    if c_datas.sub_data is not None and isinstance(c_datas.sub_data, dict):
        status = c_datas.sub_data.get(solver_status_key)
        if status is not None:
            solver_failure_rate = float(jnp.mean(status != 1))

    return SafetyStatistics(
        n_trials=n_trials,
        violation_rate=violation_rate,
        min_barrier_value=min_barrier_value,
        mean_min_barrier=mean_min_barrier,
        per_trial_min_barrier=per_trial_min,
        per_trial_violated=per_trial_violated,
        total_violation_steps=total_violation_steps,
        solver_failure_rate=solver_failure_rate,
    )
