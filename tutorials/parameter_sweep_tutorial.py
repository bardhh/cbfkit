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
    cbfkit-bench sweep configs/single_integrator/cpu/sweep_grid.yaml
    cbfkit-bench sweep configs/single_integrator/cpu/sweep_optuna.yaml
"""

import tempfile

from cbfkit.benchmarks.single_integrator_sweep import tutorial_sweep  # noqa: F401 — registers scenario
from cbfkit.benchmarks.sweep import (
    build_param_grid,
    run_sweep,
    write_sweep_artifacts,
)


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
#   cbfkit-bench sweep configs/single_integrator/cpu/sweep_grid.yaml
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
