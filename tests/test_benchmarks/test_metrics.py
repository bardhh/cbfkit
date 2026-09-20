"""Incomplete benchmark evidence must not become a successful zero measurement."""

import json

import pytest

from cbfkit.benchmarks.metrics import mean, summarize
from cbfkit.benchmarks.runner import BenchmarkRun, _parse_seeds, compare_runs, write_artifacts
from cbfkit.cli.bench import main


@pytest.fixture
def record():
    return {"success": 1, "safety_violations": 0, "solver_failures": 0, "avg_step_ms": 2.0}


@pytest.mark.parametrize("key", ["success", "safety_violations", "solver_failures", "avg_step_ms"])
def test_required_metrics_cannot_be_omitted(record, key):
    del record[key]
    with pytest.raises(ValueError, match=key):
        summarize([record])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "unknown"])
def test_invalid_measurement_is_rejected(record, value):
    record["avg_step_ms"] = value
    with pytest.raises(ValueError, match="metric value"):
        summarize([record])


def test_unavailable_measurement_is_not_zero(record, tmp_path):
    record["safety_violations"] = None
    summary = summarize([record])
    assert summary["safety_violation_rate"] is None
    run = BenchmarkRun("test", [0], [record], summary)
    write_artifacts(run, tmp_path)
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["summary"]["safety_violation_rate"] is None
    with pytest.raises(ValueError, match="unavailable"):
        compare_runs(run, run, "safety_violation_rate")


def test_optional_metrics_from_later_records_remain_unavailable(record):
    summary = summarize([record, dict(record, final_distance=3.0)])
    assert "final_distance" in summary
    assert summary["final_distance"] is None


def test_partial_measurement_is_not_averaged_over_a_subset(record):
    summary = summarize([dict(record, avg_step_ms=None), record])
    assert summary["avg_step_ms"] is None


def test_complete_measurements_average_normally(record):
    summary = summarize([record, dict(record, avg_step_ms=4.0, safety_violations=1)])
    assert summary["avg_step_ms"] == 3.0
    assert summary["safety_violation_rate"] == 0.5


def test_unknown_comparison_metric_fails(record):
    run = BenchmarkRun("test", [0], [record], summarize([record]))
    with pytest.raises(ValueError, match="avg_steps_ms"):
        compare_runs(run, run, "avg_steps_ms")
    assert compare_runs(run, run, "avg_step_ms")["delta"] == 0.0


def test_empty_summary_fails():
    with pytest.raises(ValueError, match="empty"):
        summarize([])
    with pytest.raises(ValueError, match="empty"):
        mean([], "success")


@pytest.mark.parametrize("seeds", [[], "", "3:1", "1,,2"])
def test_invalid_seed_selection_fails(seeds):
    with pytest.raises(ValueError):
        _parse_seeds(seeds)


def test_cli_reports_invalid_metric_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "compare",
                "sanity_random_safety",
                "--left-seeds",
                "0",
                "--right-seeds",
                "1",
                "--metric",
                "typo",
            ]
        )
    assert exc.value.code == 2
    assert "typo" in capsys.readouterr().err
