"""Tests for the parameter sweep engine."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cbfkit.benchmarks.registry import register_sweepable_scenario, registry
from cbfkit.benchmarks.sweep import (
    SweepRun,
    build_param_grid,
    expand_param_spec,
    run_optuna_sweep,
    run_sweep,
    sample_param_combos,
    write_sweep_artifacts,
)
from cbfkit.benchmarks.sweep_config import SweepConfigError, load_sweep_config
from cbfkit.benchmarks.sweep_viz import SweepViz


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


def _mock_failing_runner(seed: int, params: dict) -> dict:
    """Fails on seed >= 1 when alpha > 2."""
    fails = params.get("alpha", 1.0) > 2.0 and seed >= 1
    return {
        "success": 0 if fails else 1,
        "safety_violations": 1 if fails else 0,
        "solver_failures": 0,
        "avg_step_ms": 1.0,
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
# Falsifier mode
# ---------------------------------------------------------------------------


class TestFalsifier:
    def test_falsifier_skips_seeds_after_failure(self):
        """Falsifier should stop seeds for a combo on first failure."""
        combos = [{"alpha": 3.0}]  # alpha > 2 fails on seed >= 1
        seeds = [0, 1, 2, 3]
        result = run_sweep(
            "test_falsify", seeds, combos, _mock_failing_runner,
            falsifier=True, falsifier_metric="safety_violations",
        )
        # seed=0 passes, seed=1 fails -> seeds 2,3 skipped
        assert len(result.records) == 2
        assert result.per_combo_summaries[0]["falsified"] is True

    def test_falsifier_no_failure_runs_all_seeds(self):
        """When no failure occurs, all seeds run."""
        combos = [{"alpha": 1.0}]  # alpha <= 2 never fails
        seeds = [0, 1, 2]
        result = run_sweep(
            "test_falsify_pass", seeds, combos, _mock_failing_runner,
            falsifier=True,
        )
        assert len(result.records) == 3
        assert result.per_combo_summaries[0]["falsified"] is False

    def test_falsifier_disabled_runs_all(self):
        """Without falsifier, all seeds run even with failures."""
        combos = [{"alpha": 3.0}]
        seeds = [0, 1, 2, 3]
        result = run_sweep(
            "test_no_falsify", seeds, combos, _mock_failing_runner,
            falsifier=False,
        )
        assert len(result.records) == 4

    def test_falsifier_yaml_config(self):
        """YAML config should parse falsifier settings."""
        from cbfkit.benchmarks.sweep_config import load_sweep_config

        yaml_content = """\
scenario: "sanity_random_safety"
seeds: "0:2"
sweep:
  method: grid
  falsifier: true
  falsifier_metric: safety_violations
  parameters:
    alpha:
      values: [1.0, 5.0]
output:
  dir: "/tmp/test_falsify"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_sweep_config(f.name)

        assert config.falsifier is True
        assert config.falsifier_metric == "safety_violations"


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


# ---------------------------------------------------------------------------
# Optuna sweep
# ---------------------------------------------------------------------------

_has_optuna = False
try:
    import optuna  # noqa: F401

    _has_optuna = True
except ImportError:
    _has_optuna = False


@pytest.mark.skipif(not _has_optuna, reason="optuna not installed")
class TestOptunaSweep:
    def test_basic_optuna_sweep(self):
        """Optuna sweep runs the correct number of trials."""
        params = {
            "alpha": {"range": [0.1, 5.0]},
            "mode": {"values": ["fast", "slow"]},
        }
        result = run_optuna_sweep(
            "test_optuna",
            seeds=[0, 1],
            parameters=params,
            runner=_mock_runner,
            n_trials=5,
            objective_metric="avg_step_ms",
            direction="minimize",
        )
        assert isinstance(result, SweepRun)
        assert len(result.param_combos) == 5
        assert len(result.per_combo_summaries) == 5
        # 5 trials x 2 seeds = 10 records
        assert len(result.records) == 10

    def test_optuna_records_have_param_prefix(self):
        params = {"alpha": {"values": [1.0, 2.0, 3.0]}}
        result = run_optuna_sweep(
            "test_optuna_prefix",
            seeds=[0],
            parameters=params,
            runner=_mock_runner,
            n_trials=3,
            objective_metric="avg_step_ms",
        )
        for rec in result.records:
            assert "param_alpha" in rec

    def test_optuna_int_range(self):
        params = {"count": {"int_range": [1, 10]}}

        def runner(seed, p):
            return {"success": 1, "safety_violations": 0, "solver_failures": 0,
                    "avg_step_ms": float(p["count"])}

        result = run_optuna_sweep(
            "test_optuna_int",
            seeds=[0],
            parameters=params,
            runner=runner,
            n_trials=5,
            objective_metric="avg_step_ms",
        )
        for combo in result.param_combos:
            assert isinstance(combo["count"], int)
            assert 1 <= combo["count"] <= 10

    def test_optuna_writes_artifacts(self):
        params = {"alpha": {"range": [0.1, 2.0]}}
        result = run_optuna_sweep(
            "test_optuna_artifacts",
            seeds=[0],
            parameters=params,
            runner=_mock_runner,
            n_trials=3,
            objective_metric="avg_step_ms",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_sweep_artifacts(result, tmpdir)
            out = Path(tmpdir)
            assert (out / "sweep_results.json").exists()
            data = json.loads((out / "sweep_results.json").read_text())
            assert len(data["per_combo_summaries"]) == 3

    def test_optuna_falsifier(self):
        """Optuna sweep with falsifier should skip seeds after failure."""
        params = {"alpha": {"values": [1.0, 3.0, 5.0]}}
        seeds = [0, 1, 2, 3]
        result = run_optuna_sweep(
            "test_optuna_falsify",
            seeds=seeds,
            parameters=params,
            runner=_mock_failing_runner,
            n_trials=3,
            objective_metric="avg_step_ms",
            falsifier=True,
            falsifier_metric="safety_violations",
        )
        # alpha=1.0 runs all 4 seeds (no failures)
        # alpha=3.0 and alpha=5.0 fail on seed>=1, so only 2 seeds each
        # Total records should be less than 3*4=12
        assert len(result.records) < 3 * len(seeds)
        # At least one combo should be falsified
        assert any(s["falsified"] for s in result.per_combo_summaries)

    def test_optuna_config_loading(self):
        from cbfkit.benchmarks.sweep_config import load_sweep_config, resolve_param_combos

        yaml_content = """\
scenario: "sanity_random_safety"
seeds: "0:1"
sweep:
  method: optuna
  n_samples: 10
  objective: avg_step_ms
  direction: minimize
  parameters:
    alpha:
      range: [0.1, 5.0]
output:
  dir: "/tmp/test_optuna"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_sweep_config(f.name)

        assert config.method == "optuna"
        assert config.objective == "avg_step_ms"
        assert config.direction == "minimize"
        assert config.n_samples == 10
        # resolve_param_combos returns empty for optuna
        combos = resolve_param_combos(config)
        assert combos == []


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_missing_scenario_raises(self):
        yaml_content = """\
sweep:
  method: grid
  parameters:
    alpha:
      values: [1.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(SweepConfigError, match="scenario"):
                load_sweep_config(f.name)

    def test_unknown_method_raises(self):
        yaml_content = """\
scenario: "sanity_random_safety"
sweep:
  method: bayesian_magic
  parameters:
    alpha:
      values: [1.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(SweepConfigError, match="method"):
                load_sweep_config(f.name)

    def test_unknown_direction_raises(self):
        yaml_content = """\
scenario: "sanity_random_safety"
sweep:
  method: optuna
  direction: sideways
  parameters:
    alpha:
      range: [0.1, 5.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(SweepConfigError, match="direction"):
                load_sweep_config(f.name)

    def test_unknown_param_spec_raises(self):
        yaml_content = """\
scenario: "sanity_random_safety"
sweep:
  method: grid
  parameters:
    alpha:
      magic_numbers: [1.0, 2.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(SweepConfigError, match="alpha"):
                load_sweep_config(f.name)

    def test_legacy_skip_on_failure_still_works(self):
        """Backward compat: skip_on_failure maps to falsifier."""
        yaml_content = """\
scenario: "sanity_random_safety"
sweep:
  method: grid
  skip_on_failure: true
  failure_metric: solver_failures
  parameters:
    alpha:
      values: [1.0]
output:
  dir: /tmp/test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_sweep_config(f.name)
        assert config.falsifier is True
        assert config.falsifier_metric == "solver_failures"


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


class TestBatchRunner:
    def test_batch_runner_used(self):
        """When batch_runner is provided and falsifier is off, it should be used."""
        calls = []

        def batch_fn(seeds, params):
            calls.append(seeds)
            return [
                {"success": 1, "safety_violations": 0, "solver_failures": 0,
                 "avg_step_ms": 1.0, "score": params.get("alpha", 1.0) + s}
                for s in seeds
            ]

        combos = [{"alpha": 2.0}]
        result = run_sweep(
            "test_batch", [0, 1, 2], combos, _mock_runner,
            batch_runner=batch_fn,
        )
        assert len(calls) == 1  # single batched call
        assert calls[0] == [0, 1, 2]
        assert len(result.records) == 3

    def test_batch_runner_skipped_with_falsifier(self):
        """Batch runner should not be used when falsifier is on."""
        calls = []

        def batch_fn(seeds, params):
            calls.append(seeds)
            return [{"success": 1, "safety_violations": 0, "solver_failures": 0,
                     "avg_step_ms": 1.0} for _ in seeds]

        combos = [{"alpha": 3.0}]
        run_sweep(
            "test_batch_falsifier", [0, 1, 2], combos, _mock_failing_runner,
            falsifier=True, batch_runner=batch_fn,
        )
        assert len(calls) == 0  # batch runner not used


# ---------------------------------------------------------------------------
# SweepViz NaN/inf handling
# ---------------------------------------------------------------------------


class TestSweepVizNaN:
    def test_nan_values_not_tracked_as_best(self):
        viz = SweepViz(objective_metric="score", direction="minimize")
        viz.add_result({"a": 1}, {"score": float("inf")})
        viz.add_result({"a": 2}, {"score": 5.0})
        viz.add_result({"a": 3}, {"score": float("nan")})
        viz.add_result({"a": 4}, {"score": 3.0})

        assert viz._best_val == 3.0
        assert viz._best_idx == 3

    def test_first_finite_becomes_best(self):
        viz = SweepViz(objective_metric="score", direction="maximize")
        viz.add_result({"a": 1}, {"score": float("inf")})
        viz.add_result({"a": 2}, {"score": 10.0})
        assert viz._best_val == 10.0
        assert viz._best_idx == 1

    def test_all_inf_no_best(self):
        viz = SweepViz(objective_metric="score", direction="minimize")
        viz.add_result({"a": 1}, {"score": float("inf")})
        viz.add_result({"a": 2}, {"score": float("inf")})
        assert viz._best_idx == -1
        assert not math.isfinite(viz._best_val)


# ---------------------------------------------------------------------------
# Obstacle config parsing
# ---------------------------------------------------------------------------


class TestObstacleConfig:
    def test_parse_circular_fixed(self):
        yaml_content = """\
scenario: "tutorial_single_integrator_sweep"
seeds: "0:1"
obstacles:
  type: circular
  items:
    - center: [3.0, 4.0]
      radius: 0.8
    - center: [6.0, 5.0]
      radius: 1.2
sweep:
  method: grid
  parameters:
    alpha:
      values: [1.0, 2.0]
output:
  dir: /tmp/test_obs
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_sweep_config(f.name)

        assert config.obstacles is not None
        assert config.obstacles.type == "circular"
        assert len(config.obstacles.items) == 2
        # Both items fully fixed
        assert config.obstacles.items[0].fixed == {"center": [3.0, 4.0], "radius": 0.8}
        assert config.obstacles.items[0].sweepable == {}
        # No synthetic params added
        assert "obstacle_0_radius" not in config.parameters

    def test_parse_circular_sweepable_radius(self):
        yaml_content = """\
scenario: "tutorial_single_integrator_sweep"
seeds: "0:0"
obstacles:
  type: circular
  items:
    - center: [3.0, 4.0]
      radius:
        linspace: [0.3, 1.5, 5]
sweep:
  method: grid
  parameters:
    alpha:
      values: [1.0]
output:
  dir: /tmp/test_obs
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_sweep_config(f.name)

        assert config.obstacles is not None
        assert config.obstacles.items[0].sweepable == {
            "radius": {"linspace": [0.3, 1.5, 5]}
        }
        # Synthetic param merged into parameters
        assert "obstacle_0_radius" in config.parameters
        assert config.parameters["obstacle_0_radius"] == {"linspace": [0.3, 1.5, 5]}

    def test_parse_ellipsoidal_fixed(self):
        yaml_content = """\
scenario: "unicycle_obstacle_avoidance_sweep"
seeds: "0:0"
obstacles:
  type: ellipsoidal
  items:
    - center: [1.0, 2.0, 0.0]
      semi_axes: [0.5, 1.5]
sweep:
  method: grid
  parameters:
    alpha:
      values: [1.0]
output:
  dir: /tmp/test_obs
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_sweep_config(f.name)

        assert config.obstacles is not None
        assert config.obstacles.type == "ellipsoidal"
        assert config.obstacles.items[0].fixed == {
            "center": [1.0, 2.0, 0.0],
            "semi_axes": [0.5, 1.5],
        }

    def test_invalid_obstacle_type_raises(self):
        yaml_content = """\
scenario: "tutorial_single_integrator_sweep"
seeds: "0:0"
obstacles:
  type: hexagonal
  items:
    - center: [1.0, 2.0]
      radius: 0.5
sweep:
  method: grid
  parameters:
    alpha:
      values: [1.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(SweepConfigError, match="obstacles.type"):
                load_sweep_config(f.name)

    def test_missing_required_key_raises(self):
        yaml_content = """\
scenario: "tutorial_single_integrator_sweep"
seeds: "0:0"
obstacles:
  type: circular
  items:
    - center: [1.0, 2.0]
sweep:
  method: grid
  parameters:
    alpha:
      values: [1.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(SweepConfigError, match="missing required keys"):
                load_sweep_config(f.name)

    def test_no_obstacles_block_returns_none(self):
        yaml_content = """\
scenario: "tutorial_single_integrator_sweep"
seeds: "0:0"
sweep:
  method: grid
  parameters:
    alpha:
      values: [1.0]
output:
  dir: /tmp/test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_sweep_config(f.name)

        assert config.obstacles is None

    def test_build_obstacle_fixed_params(self):
        from cbfkit.benchmarks.sweep_config import ObstacleItemSpec, ObstaclesSpec, _build_obstacle_fixed_params

        spec = ObstaclesSpec(
            type="circular",
            items=[
                ObstacleItemSpec(
                    fixed={"center": [3.0, 4.0], "radius": 0.8},
                    sweepable={},
                ),
                ObstacleItemSpec(
                    fixed={"center": [6.0, 5.0]},
                    sweepable={"radius": {"linspace": [0.3, 1.5, 5]}},
                ),
            ],
        )
        fixed = _build_obstacle_fixed_params(spec)
        assert fixed["_obstacle_count"] == 2
        assert fixed["_obstacle_type"] == "circular"
        assert fixed["_obstacle_0_center"] == [3.0, 4.0]
        assert fixed["_obstacle_0_radius"] == 0.8
        assert fixed["_obstacle_1_center"] == [6.0, 5.0]
        # Sweepable radius is NOT in fixed params
        assert "_obstacle_1_radius" not in fixed

    def test_extract_obstacle_sweep_params(self):
        from cbfkit.benchmarks.sweep_config import ObstacleItemSpec, ObstaclesSpec, _extract_obstacle_sweep_params

        spec = ObstaclesSpec(
            type="circular",
            items=[
                ObstacleItemSpec(fixed={"center": [3.0, 4.0]}, sweepable={"radius": {"values": [0.5, 1.0]}}),
                ObstacleItemSpec(fixed={"center": [6.0, 5.0], "radius": 1.2}, sweepable={}),
            ],
        )
        sweep_params = _extract_obstacle_sweep_params(spec)
        assert sweep_params == {"obstacle_0_radius": {"values": [0.5, 1.0]}}


# ---------------------------------------------------------------------------
# Obstacle resolution
# ---------------------------------------------------------------------------


class TestObstacleResolution:
    def test_resolve_circular_fixed_only(self):
        from cbfkit.benchmarks.scenario_builders import resolve_circular_obstacles

        params = {
            "_obstacle_count": 2,
            "_obstacle_type": "circular",
            "_obstacle_0_center": [3.0, 4.0],
            "_obstacle_0_radius": 0.8,
            "_obstacle_1_center": [6.0, 5.0],
            "_obstacle_1_radius": 1.2,
        }
        result = resolve_circular_obstacles(params)
        assert result is not None
        assert len(result) == 2
        np.testing.assert_allclose(result[0][0], [3.0, 4.0])
        assert result[0][1] == 0.8
        np.testing.assert_allclose(result[1][0], [6.0, 5.0])
        assert result[1][1] == 1.2

    def test_resolve_circular_with_swept_radius(self):
        from cbfkit.benchmarks.scenario_builders import resolve_circular_obstacles

        # Swept values use plain keys; fixed use underscore-prefixed
        params = {
            "_obstacle_count": 2,
            "_obstacle_type": "circular",
            "_obstacle_0_center": [3.0, 4.0],
            "obstacle_0_radius": 0.5,  # swept value
            "_obstacle_1_center": [6.0, 5.0],
            "_obstacle_1_radius": 1.2,  # fixed value
        }
        result = resolve_circular_obstacles(params)
        assert result is not None
        assert result[0][1] == 0.5  # swept radius
        assert result[1][1] == 1.2  # fixed radius

    def test_resolve_returns_none_when_no_obstacles(self):
        from cbfkit.benchmarks.scenario_builders import resolve_circular_obstacles

        assert resolve_circular_obstacles({"alpha": 1.0}) is None

    def test_resolve_ellipsoidal_fixed(self):
        from cbfkit.benchmarks.scenario_builders import resolve_ellipsoidal_obstacles

        params = {
            "_obstacle_count": 1,
            "_obstacle_type": "ellipsoidal",
            "_obstacle_0_center": [1.0, 2.0, 0.0],
            "_obstacle_0_semi_axes": [0.5, 1.5],
        }
        result = resolve_ellipsoidal_obstacles(params)
        assert result is not None
        centers, semi_axes = result
        assert len(centers) == 1
        np.testing.assert_allclose(centers[0], [1.0, 2.0, 0.0])
        np.testing.assert_allclose(semi_axes[0], [0.5, 1.5])

    def test_resolve_ellipsoidal_returns_none_when_no_obstacles(self):
        from cbfkit.benchmarks.scenario_builders import resolve_ellipsoidal_obstacles

        assert resolve_ellipsoidal_obstacles({"alpha": 1.0}) is None
