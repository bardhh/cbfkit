"""Tests for the parameter sweep engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cbfkit.benchmarks.registry import register_sweepable_scenario, registry
from cbfkit.benchmarks.sweep import (
    SweepRun,
    build_param_grid,
    expand_param_spec,
    run_sweep,
    sample_param_combos,
    write_sweep_artifacts,
)


# ---------------------------------------------------------------------------
# expand_param_spec
# ---------------------------------------------------------------------------


class TestExpandParamSpec:
    def test_values(self):
        result = expand_param_spec({"values": [1, 2, 3]})
        assert result == [1, 2, 3]

    def test_values_bool(self):
        result = expand_param_spec({"values": [True, False]})
        assert result == [True, False]

    def test_linspace(self):
        result = expand_param_spec({"linspace": [0.0, 1.0, 3]})
        assert len(result) == 3
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_logspace(self):
        result = expand_param_spec({"logspace": [0, 2, 3]})
        assert len(result) == 3
        np.testing.assert_allclose(result, [1.0, 10.0, 100.0])

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            expand_param_spec({"something_else": [1, 2]})


# ---------------------------------------------------------------------------
# build_param_grid
# ---------------------------------------------------------------------------


class TestBuildParamGrid:
    def test_single_param(self):
        grid = build_param_grid({"a": {"values": [1, 2, 3]}})
        assert len(grid) == 3
        assert grid[0] == {"a": 1}
        assert grid[2] == {"a": 3}

    def test_cartesian_product(self):
        grid = build_param_grid({
            "x": {"values": [1, 2]},
            "y": {"values": [10, 20]},
        })
        assert len(grid) == 4
        assert {"x": 1, "y": 10} in grid
        assert {"x": 2, "y": 20} in grid

    def test_empty(self):
        grid = build_param_grid({})
        assert grid == [{}]


# ---------------------------------------------------------------------------
# sample_param_combos
# ---------------------------------------------------------------------------


class TestSampleParamCombos:
    def test_fewer_than_grid(self):
        params = {"a": {"values": [1, 2, 3, 4, 5]}}
        combos = sample_param_combos(params, n_samples=2, seed=42)
        assert len(combos) == 2

    def test_more_than_grid_returns_full(self):
        params = {"a": {"values": [1, 2]}}
        combos = sample_param_combos(params, n_samples=100, seed=0)
        assert len(combos) == 2

    def test_deterministic(self):
        params = {"a": {"values": list(range(20))}}
        c1 = sample_param_combos(params, n_samples=5, seed=7)
        c2 = sample_param_combos(params, n_samples=5, seed=7)
        assert c1 == c2


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------


def _mock_runner(seed: int, params: dict) -> dict:
    return {
        "success": 1,
        "safety_violations": 0,
        "solver_failures": 0,
        "avg_step_ms": params.get("alpha", 1.0) * 0.1 + seed * 0.01,
        "score": params.get("alpha", 1.0) + seed,
    }


class TestRunSweep:
    def test_basic_sweep(self):
        combos = [{"alpha": 1.0}, {"alpha": 2.0}]
        result = run_sweep("test", [0, 1], combos, _mock_runner)

        assert isinstance(result, SweepRun)
        assert len(result.records) == 4  # 2 combos x 2 seeds
        assert len(result.per_combo_summaries) == 2
        assert result.records[0]["param_alpha"] == 1.0

    def test_summary_aggregation(self):
        combos = [{"alpha": 5.0}]
        result = run_sweep("test", [0, 1, 2], combos, _mock_runner)
        summary = result.per_combo_summaries[0]

        assert summary["num_runs"] == 3.0
        assert summary["param_alpha"] == 5.0


# ---------------------------------------------------------------------------
# write_sweep_artifacts
# ---------------------------------------------------------------------------


class TestWriteSweepArtifacts:
    def test_writes_files(self):
        combos = [{"alpha": 1.0}]
        result = run_sweep("test", [0], combos, _mock_runner)

        with tempfile.TemporaryDirectory() as tmpdir:
            write_sweep_artifacts(result, tmpdir)
            out = Path(tmpdir)

            assert (out / "sweep_results.json").exists()
            assert (out / "sweep_records.csv").exists()
            assert (out / "sweep_summary.csv").exists()

            data = json.loads((out / "sweep_results.json").read_text())
            assert data["scenario"] == "test"
            assert len(data["records"]) == 1


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------


class TestSweepConfig:
    def test_load_config(self):
        from cbfkit.benchmarks.sweep_config import load_sweep_config

        yaml_content = """\
scenario: "sanity_random_safety"
seeds: "0:2"
sweep:
  method: grid
  parameters:
    alpha:
      values: [1.0, 2.0]
output:
  dir: "/tmp/test_sweep"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_sweep_config(f.name)

        assert config.scenario == "sanity_random_safety"
        assert config.seeds == [0, 1, 2]
        assert config.method == "grid"
        assert "alpha" in config.parameters

    def test_resolve_grid(self):
        from cbfkit.benchmarks.sweep_config import SweepConfig, resolve_param_combos

        config = SweepConfig(
            scenario="test",
            seeds=[0],
            method="grid",
            parameters={"a": {"values": [1, 2]}, "b": {"values": [10, 20]}},
            n_samples=50,
            output_dir="/tmp",
        )
        combos = resolve_param_combos(config)
        assert len(combos) == 4

    def test_resolve_random(self):
        from cbfkit.benchmarks.sweep_config import SweepConfig, resolve_param_combos

        config = SweepConfig(
            scenario="test",
            seeds=[0],
            method="random",
            parameters={"a": {"values": list(range(10))}},
            n_samples=3,
            output_dir="/tmp",
        )
        combos = resolve_param_combos(config)
        assert len(combos) == 3


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_sweepable_scenario_registered(self):
        """The monte_carlo_gpu_sweep scenario should be registered."""
        spec = registry.scenario("monte_carlo_gpu_sweep")
        assert spec.sweep_runner is not None
        assert "alpha" in spec.sweepable_params

    def test_default_runner_works(self):
        """The auto-generated default runner (no params) should work."""
        spec = registry.scenario("sanity_random_safety")
        result = spec.runner(0)
        assert "success" in result

    def test_list_shows_sweepable(self):
        spec = registry.scenario("monte_carlo_gpu_sweep")
        assert len(spec.sweepable_params) > 0
