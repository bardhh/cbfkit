"""
Parameter Sweep Tutorial: Tuning CBF Controllers with Grid Search & Optuna
===========================================================================

This tutorial shows how to make *any* cbfkit controller sweepable so you can
systematically find optimal parameters. We use a 2D single integrator
navigating toward a goal while avoiding circular obstacles — the same system
used in the GPU Monte Carlo benchmark.

Three sweep methods are demonstrated:
  1. **Grid sweep** — exhaustive search over a parameter grid
  2. **Random sweep** — random subset of the grid
  3. **Optuna sweep** — Bayesian optimization (TPE) for efficient search

Usage::

    # Run this file directly to see the sweep in action:
    python tutorials/parameter_sweep_tutorial.py

    # Or use the CLI with YAML configs:
    cbfkit-bench sweep configs/tutorial_sweep_grid.yaml
    cbfkit-bench sweep configs/tutorial_sweep_optuna.yaml
"""

import json
import time
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random

from cbfkit.benchmarks.registry import register_sweepable_scenario
from cbfkit.benchmarks.sweep import (
    build_param_grid,
    run_sweep,
    write_sweep_artifacts,
)
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_vanilla_clf_constraints,
    generate_compute_zeroing_cbf_constraints,
)
from cbfkit.integration import forward_euler
from cbfkit.simulation.monte_carlo_gpu import MonteCarloSetup, conduct_monte_carlo_gpu
from cbfkit.simulation.safety_verification import compute_safety_statistics
from cbfkit.utils.user_types import CertificateCollection, ControllerData, PlannerData


# ============================================================================
# Step 1: Define your system
# ============================================================================
# A 2D single integrator: x_dot = u
# Goal: reach [10, 10] while avoiding circular obstacles.


def _dynamics(x):
    """Single integrator: f(x) = 0, g(x) = I."""
    return jnp.zeros(2), jnp.eye(2)


def _make_cbf(center, radius, alpha):
    """Circular obstacle barrier: h(x) = |x - c|^2 - r^2."""
    return (
        lambda _t, x: jnp.sum((x - center) ** 2) - radius**2,
        lambda _t, x: 2 * (x - center),
        lambda _t, _x: 2 * jnp.eye(2),
        lambda _t, _x: 0.0,
        lambda h, _a=alpha: _a * h,  # Class K: alpha * h
    )


def _default_sensor(t, x, *, sigma=None, key=None):
    return x


def _default_estimator(t, y, z, u, c):
    return y, c if c is not None else jnp.zeros((len(y), len(y)))


def _default_perturbation(x, u, f, g):
    def p(key):
        return jnp.zeros_like(x)
    return p


# ============================================================================
# Step 2: Wrap your simulation in a sweepable scenario
# ============================================================================
#
# The key idea: extract the parameters you want to tune into a `params` dict.
# The function signature must be:
#
#     def my_scenario(seed: int, params: dict) -> dict
#
# Return a dict with at least these standard metric keys:
#   - success (0 or 1)
#   - safety_violations (0 or 1)
#   - solver_failures (0 or 1)
#   - avg_step_ms (float)
# Plus any custom metrics you care about (e.g., violation_rate, min_barrier).


N_TRIALS = 10  # Monte Carlo trajectories per evaluation (increase for production)


@register_sweepable_scenario(
    "tutorial_single_integrator_sweep",
    sweepable_params=["alpha", "control_limit", "n_obstacles"],
    description="Single integrator obstacle avoidance with sweepable CBF parameters.",
)
def tutorial_sweep(seed: int, params: dict) -> dict:
    """Run Monte Carlo simulations with the given CBF parameters and return metrics."""

    # --- Extract sweep parameters (with sensible defaults) ---
    alpha = params.get("alpha", 1.0)
    control_limit = params.get("control_limit", 5.0)
    n_obstacles = int(params.get("n_obstacles", 5))

    # --- Generate random obstacles from the seed ---
    key = random.PRNGKey(seed)
    key_c, key_r = random.split(key)
    centers = random.uniform(key_c, (n_obstacles, 2), minval=2.0, maxval=8.0)
    radii = random.uniform(key_r, (n_obstacles,), minval=0.5, maxval=1.0)

    # --- Build CBF barriers with the swept alpha ---
    barrier_tuples = [_make_cbf(centers[i], radii[i], alpha) for i in range(n_obstacles)]
    barriers = CertificateCollection(*[list(x) for x in zip(*barrier_tuples)])

    # --- Create controller with swept control limits ---
    controller = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints,
    )(
        control_limits=jnp.array([control_limit, control_limit]),
        dynamics_func=_dynamics,
        barriers=barriers,
        relaxable_cbf=False,
        relaxable_clf=True,
    )

    def nominal_controller(t, x, _k, _r):
        return 2.0 * (jnp.array([10.0, 10.0]) - x), None

    def initial_state_sampler(key):
        return random.uniform(key, (2,), minval=-1.0, maxval=1.0)

    # Prime controller data
    prime_key = random.PRNGKey(42)
    _, c_data = controller(0.0, jnp.zeros(2), jnp.zeros(2), prime_key, ControllerData())

    setup = MonteCarloSetup(
        dt=0.01, num_steps=200,
        dynamics=_dynamics, integrator=forward_euler,
        initial_state_sampler=initial_state_sampler,
        nominal_controller=nominal_controller, controller=controller,
        sensor=_default_sensor, estimator=_default_estimator,
        perturbation=_default_perturbation, sigma=jnp.zeros(0),
        controller_data=c_data, planner=None, planner_data=PlannerData(),
    )

    # --- Run Monte Carlo ---
    results = conduct_monte_carlo_gpu(setup, n_trials=N_TRIALS, seed=seed)
    stats = compute_safety_statistics(results)

    return {
        "success": 1,
        "safety_violations": int(stats.violation_rate > 0),
        "solver_failures": 0,
        "avg_step_ms": results.wall_time_s / setup.num_steps * 1000.0,
        "violation_rate": stats.violation_rate,
        "min_barrier_value": stats.min_barrier_value,
        "wall_time_s": results.wall_time_s,
    }


# ============================================================================
# Step 3: Run a grid sweep programmatically
# ============================================================================

def demo_grid_sweep():
    """Run a small grid sweep and print results."""
    print("\n=== Grid Sweep: alpha vs safety_violation_rate ===")

    param_combos = build_param_grid({
        "alpha": {"values": [0.5, 1.0, 5.0]},
        "control_limit": {"values": [3.0, 10.0]},
    })
    print(f"Grid: {len(param_combos)} parameter combinations x 1 seed\n")

    result = run_sweep(
        "tutorial_single_integrator_sweep",
        seeds=[0],
        param_combos=param_combos,
        runner=tutorial_sweep,
    )

    # Print results table
    print(f"\n{'alpha':>8} {'ctrl_lim':>10} {'viol_rate':>10} {'min_barrier':>12} {'time_s':>8}")
    print("-" * 52)
    for rec in result.records:
        print(
            f"{rec['param_alpha']:8.1f} "
            f"{rec['param_control_limit']:10.1f} "
            f"{rec['violation_rate']:10.3f} "
            f"{rec['min_barrier_value']:12.3f} "
            f"{rec['wall_time_s']:8.2f}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        write_sweep_artifacts(result, tmpdir)
        print(f"\nArtifacts saved to {tmpdir}/")


# ============================================================================
# Step 4: Run an Optuna sweep programmatically
# ============================================================================

def demo_optuna_sweep():
    """Run Optuna to find the alpha that minimizes safety violations."""
    try:
        from cbfkit.benchmarks.sweep import run_optuna_sweep
    except ImportError:
        print("\nSkipping Optuna demo (install with: pip install cbfkit[optuna])")
        return

    print("\n=== Optuna Sweep: minimize safety_violation_rate ===")

    result = run_optuna_sweep(
        "tutorial_single_integrator_sweep",
        seeds=[0],
        parameters={
            "alpha": {"range": [0.1, 10.0]},
            "control_limit": {"range": [1.0, 15.0]},
        },
        runner=tutorial_sweep,
        n_trials=10,
        objective_metric="safety_violation_rate",
        direction="minimize",
    )

    # Find best trial
    best = min(result.records, key=lambda r: r["violation_rate"])
    print(f"\nBest found: alpha={best['param_alpha']:.2f}, "
          f"ctrl_lim={best['param_control_limit']:.1f}, "
          f"violation_rate={best['violation_rate']:.3f}")


# ============================================================================
# Step 5: Use via CLI with YAML config files
# ============================================================================
#
# Create a YAML config and run with the CLI:
#
#   cbfkit-bench sweep configs/tutorial_sweep_grid.yaml
#
# Then visualize:
#
#   cbfkit-bench sweep-plot ./results/tutorial_grid/sweep_results.json \
#       --x-param alpha --y-metric safety_violation_rate \
#       --hue control_limit --kind line
#
# To register your scenario for CLI use, either:
#   a) Import this module in src/cbfkit/benchmarks/__init__.py, OR
#   b) Add a small registration script that imports your scenario


if __name__ == "__main__":
    demo_grid_sweep()
    demo_optuna_sweep()
