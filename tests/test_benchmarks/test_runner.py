import json
from pathlib import Path

from cbfkit.benchmarks import registry
from cbfkit.benchmarks.runner import _parse_seeds, run_scenario, write_artifacts
from cbfkit.cli.bench import main


def test_parse_seed_range():
    assert _parse_seeds("2:4") == [2, 3, 4]


def test_qp_solver_stress_scenario_registered():
    assert "qp_solver_stress" in registry.names()


def test_run_scenario_summary_has_expected_keys():
    run = run_scenario("sanity_random_safety", "0:2")
    assert run.summary["num_runs"] == 3.0
    assert "success_rate" in run.summary
    assert len(run.records) == 3


def test_write_artifacts(tmp_path: Path):
    run = run_scenario("sanity_random_safety", [0, 1])
    write_artifacts(run, tmp_path)

    results = json.loads((tmp_path / "results.json").read_text())
    assert results["scenario"] == "sanity_random_safety"
    assert (tmp_path / "records.csv").exists()


def test_cli_run_writes_output(tmp_path: Path):
    rc = main(["run", "sanity_random_safety", "--seeds", "0:1", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "results.json").exists()
