"""Sweep decisions require complete evidence and a nonempty experiment."""

import pytest
from rich.console import Console

from cbfkit.benchmarks.sweep import run_optuna_sweep, run_sweep
from cbfkit.benchmarks.sweep_viz import SweepViz


def runner(seed, params):
    return {"success": 1, "safety_violations": 0, "solver_failures": 0, "avg_step_ms": 1.0}


@pytest.mark.parametrize("seeds,combos", [([], [{}]), ([0], [])])
def test_empty_grid_sweep_is_rejected(seeds, combos):
    with pytest.raises(ValueError, match="at least one"):
        run_sweep("test", seeds, combos, runner)


def test_missing_falsifier_metric_is_rejected():
    with pytest.raises(ValueError, match="typo"):
        run_sweep("test", [0], [{}], runner, falsifier=True, falsifier_metric="typo")


@pytest.mark.parametrize("count", [0, 2])
def test_batch_result_count_must_match_seeds(count):
    with pytest.raises(ValueError, match="one result per requested seed"):
        run_sweep(
            "test", [0], [{}], runner, batch_runner=lambda seeds, params: [runner(0, {})] * count
        )


@pytest.mark.parametrize(
    "kwargs", [{"objective_metric": "typo"}, {"safety_constraint": ("typo", 0)}]
)
def test_optuna_cannot_optimize_or_accept_missing_metrics(kwargs):
    pytest.importorskip("optuna")
    with pytest.raises(ValueError, match="typo"):
        run_optuna_sweep("test", [0], {}, runner, n_trials=1, **kwargs)


@pytest.mark.parametrize("seeds,trials", [([], 1), ([0], 0)])
def test_empty_optuna_sweep_is_rejected(seeds, trials):
    with pytest.raises(ValueError, match="at least one"):
        run_optuna_sweep("test", seeds, {}, runner, n_trials=trials)


def test_sweep_display_does_not_label_unknown_safety_as_safe():
    viz = SweepViz(objective_metric="avg_step_ms")
    viz.add_result({}, {"avg_step_ms": 1.0, "safety_violation_rate": None})
    console = Console(width=100)
    with console.capture() as capture:
        console.print(viz.render())
    assert "N/A" in capture.get()
    assert viz._safe_count == 0


def test_sweep_display_uses_aggregated_violation_rate():
    viz = SweepViz(objective_metric="avg_step_ms")
    viz.add_result({}, {"avg_step_ms": 1.0, "safety_violation_rate": 0.5})
    assert viz._safe_count == 0
    console = Console(width=100)
    with console.capture() as capture:
        console.print(viz.render())
    assert "0.5" in capture.get()


def test_sweep_display_rejects_unknown_objective():
    viz = SweepViz(objective_metric="typo")
    with pytest.raises(ValueError, match="typo"):
        viz.add_result({}, {"avg_step_ms": 1.0})
    assert not viz.trials
