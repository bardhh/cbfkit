"""Parameter sweep engine for benchmark scenarios."""

from __future__ import annotations

__all__ = [
    "SweepRun",
    "build_param_grid",
    "expand_param_spec",
    "run_optuna_sweep",
    "run_sweep",
    "sample_param_combos",
    "write_sweep_artifacts",
]

import contextlib
import csv
import itertools
import json
import os
import random as pyrandom
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Group
from rich.live import Live
from rich.progress import Progress

from ._progress import console as _console, make_progress as _make_progress
from .metrics import summarize
from .registry import BatchSweepableRunner, SweepableRunner
from .sweep_viz import SweepViz


@contextlib.contextmanager
def _quiet_stdout():
    """Redirect stdout to /dev/null to suppress jax.debug.print noise."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


# ---------------------------------------------------------------------------
# Parameter space helpers
# ---------------------------------------------------------------------------


def expand_param_spec(spec: dict[str, Any]) -> list[Any]:
    """Expand a single parameter specification into a list of values.

    Supported specs:
        ``{"values": [1, 2, 3]}``
        ``{"linspace": [start, stop, num]}``
        ``{"logspace": [start, stop, num]}``   (powers of 10)
    """
    if "values" in spec:
        return list(spec["values"])
    if "linspace" in spec:
        start, stop, num = spec["linspace"]
        return [float(v) for v in np.linspace(start, stop, int(num))]
    if "logspace" in spec:
        start, stop, num = spec["logspace"]
        return [float(v) for v in np.logspace(start, stop, int(num))]
    raise ValueError(f"Unknown param spec keys: {set(spec.keys())}")


def build_param_grid(parameters: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Cartesian product of all parameter value lists."""
    names = list(parameters.keys())
    value_lists = [expand_param_spec(parameters[n]) for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def sample_param_combos(
    parameters: dict[str, dict[str, Any]],
    n_samples: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Randomly sample *n_samples* combos from the full grid."""
    full_grid = build_param_grid(parameters)
    if n_samples >= len(full_grid):
        return full_grid
    rng = pyrandom.Random(seed)
    return rng.sample(full_grid, n_samples)


# ---------------------------------------------------------------------------
# Sweep result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepRun:
    scenario: str
    seeds: list[int]
    param_combos: list[dict[str, Any]]
    records: list[dict[str, Any]]
    per_combo_summaries: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------------


def _resolve_falsifier_kwargs(
    falsifier: bool,
    falsifier_metric: str,
    skip_on_failure: bool | None,
    failure_metric: str | None,
) -> tuple[bool, str]:
    """Resolve legacy alias kwargs to canonical names."""
    if skip_on_failure is not None:
        falsifier = skip_on_failure
    if failure_metric is not None:
        falsifier_metric = failure_metric
    return falsifier, falsifier_metric


def _is_failure(result: dict[str, Any], metric: str) -> bool:
    """Check if a result indicates a failure."""
    val = result.get(metric, 0)
    if isinstance(val, bool):
        return val
    return float(val) > 0


def _run_combo(
    runner: SweepableRunner,
    seeds: list[int],
    combo: dict[str, Any],
    combo_idx: int,
    records: list[dict[str, Any]],
    *,
    falsifier: bool = False,
    falsifier_metric: str = "safety_violations",
    progress: Progress | None = None,
    seed_task_id: int | None = None,
    batch_runner: BatchSweepableRunner | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Run a single parameter combo across seeds.

    When *batch_runner* is provided and *falsifier* is False, all
    seeds are executed in a single batched call (one JIT compilation).

    When *falsifier* is True, stops iterating seeds as soon as a
    failure is detected and moves on to the next combo.

    Returns ``(combo_records, falsified)`` where *combo_records* contains
    only the seeds that actually executed and *falsified* indicates whether
    remaining seeds were skipped due to failure.
    """
    combo_records: list[dict[str, Any]] = []
    was_falsified = False

    # Use batch runner when available and falsifier is off
    if batch_runner is not None and not falsifier:
        batch_results = batch_runner(seeds, combo)
        for seed, result in zip(seeds, batch_results):
            result = dict(result)
            result["seed"] = seed
            result["combo_idx"] = combo_idx
            for k, v in combo.items():
                result[f"param_{k}"] = v
            records.append(result)
            combo_records.append(result)
        if progress is not None and seed_task_id is not None:
            progress.advance(seed_task_id, len(seeds))
        return combo_records, False

    for seed in seeds:
        result = dict(runner(seed, combo))
        result["seed"] = seed
        result["combo_idx"] = combo_idx
        for k, v in combo.items():
            result[f"param_{k}"] = v

        if falsifier and _is_failure(result, falsifier_metric):
            result["skipped_remaining"] = True
            was_falsified = True

        records.append(result)
        combo_records.append(result)

        if progress is not None and seed_task_id is not None:
            progress.advance(seed_task_id, 1)

        if was_falsified:
            break

    return combo_records, was_falsified


def _build_combo_summary(
    combo_records: list[dict[str, Any]],
    combo: dict[str, Any],
    combo_idx: int,
    falsified: bool,
) -> dict[str, Any]:
    """Create an aggregated summary for a single parameter combo."""
    summary = summarize(combo_records)
    summary["combo_idx"] = combo_idx
    summary["falsified"] = falsified
    for k, v in combo.items():
        summary[f"param_{k}"] = v
    return summary


def _format_combo_desc(combo: dict[str, Any], max_len: int = 40) -> str:
    """Format a parameter combo as a short description string."""
    desc = ", ".join(
        f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in combo.items()
    )
    if len(desc) > max_len:
        desc = desc[: max_len - 3] + "..."
    return desc


def _process_combo(
    runner: SweepableRunner,
    seeds: list[int],
    combo: dict[str, Any],
    combo_idx: int,
    records: list[dict[str, Any]],
    per_combo_summaries: list[dict[str, Any]],
    *,
    falsifier: bool,
    falsifier_metric: str,
    progress: Progress,
    seed_task_id: int,
    combo_task_id: int,
    batch_runner: BatchSweepableRunner | None,
    viz: SweepViz | None,
    live: Live | None,
    live_renderable_fn: Any = None,
) -> int:
    """Run a single combo, update progress/viz, return number of skipped seeds."""
    progress.reset(seed_task_id, total=len(seeds))
    combo_desc = _format_combo_desc(combo)
    progress.update(seed_task_id, description=f"  Seeds ({combo_desc})")

    combo_records, was_falsified = _run_combo(
        runner, seeds, combo, combo_idx, records,
        falsifier=falsifier, falsifier_metric=falsifier_metric,
        progress=progress, seed_task_id=seed_task_id,
        batch_runner=batch_runner,
    )

    remaining = len(seeds) - len(combo_records)
    if remaining > 0:
        progress.advance(seed_task_id, remaining)

    progress.advance(combo_task_id, 1)

    summary = _build_combo_summary(combo_records, combo, combo_idx, was_falsified)
    per_combo_summaries.append(summary)

    if viz is not None and live is not None and live_renderable_fn is not None:
        viz.add_result(combo, summary)
        live.update(live_renderable_fn())

    return remaining


def run_sweep(
    name: str,
    seeds: list[int],
    param_combos: list[dict[str, Any]],
    runner: SweepableRunner,
    *,
    falsifier: bool = False,
    falsifier_metric: str = "safety_violations",
    viz: SweepViz | None = None,
    batch_runner: BatchSweepableRunner | None = None,
    # Legacy aliases
    skip_on_failure: bool | None = None,
    failure_metric: str | None = None,
) -> SweepRun:
    """Execute a scenario across all (seed, param_combo) pairs.

    Parameters
    ----------
    falsifier : bool
        When *True*, stop iterating seeds for a parameter combo as soon as
        a failure is detected (the metric given by *falsifier_metric* is
        non-zero) and move on to the next combo.
    falsifier_metric : str
        The result key checked for failure (default ``"safety_violations"``).
    viz : SweepViz | None
        Optional live visualization.  When provided the sweep renders a
        colour-coded results table and scatter plot in the terminal.
    """
    falsifier, falsifier_metric = _resolve_falsifier_kwargs(
        falsifier, falsifier_metric, skip_on_failure, failure_metric,
    )

    records: list[dict[str, Any]] = []
    per_combo_summaries: list[dict[str, Any]] = []
    skipped = 0

    progress = _make_progress()

    def _build_live_renderable():
        if viz is not None:
            return Group(viz.render_header(), progress, viz.render())
        return progress

    with Live(_build_live_renderable(), console=_console, refresh_per_second=4,
              transient=True, vertical_overflow="visible") as live, \
            _quiet_stdout():
        combo_task = progress.add_task("Combos", total=len(param_combos))
        seed_task = progress.add_task("  Seeds", total=len(seeds))

        for combo_idx, combo in enumerate(param_combos):
            remaining = _process_combo(
                runner, seeds, combo, combo_idx, records, per_combo_summaries,
                falsifier=falsifier, falsifier_metric=falsifier_metric,
                progress=progress, seed_task_id=seed_task, combo_task_id=combo_task,
                batch_runner=batch_runner, viz=viz, live=live,
                live_renderable_fn=_build_live_renderable,
            )
            skipped += remaining

    if skipped > 0:
        _console.print(
            f"[dim]Skipped {skipped} runs (moved to next combo on failure)[/dim]"
        )

    # Print final viz so it persists after Live exits
    if viz is not None:
        _console.print(viz.render_final())

    return SweepRun(
        scenario=name,
        seeds=seeds,
        param_combos=param_combos,
        records=records,
        per_combo_summaries=per_combo_summaries,
    )


# ---------------------------------------------------------------------------
# Optuna-based sweep
# ---------------------------------------------------------------------------


def _suggest_param(trial, name: str, spec: dict[str, Any]) -> Any:
    """Map a YAML param spec to an Optuna suggestion."""
    if "values" in spec:
        return trial.suggest_categorical(name, spec["values"])
    if "range" in spec:
        low, high = spec["range"]
        log = spec.get("log", False)
        return trial.suggest_float(name, low, high, log=log)
    if "int_range" in spec:
        low, high = spec["int_range"]
        return trial.suggest_int(name, low, high)
    if "linspace" in spec:
        start, stop, num = spec["linspace"]
        vals = [float(v) for v in np.linspace(start, stop, int(num))]
        return trial.suggest_categorical(name, vals)
    if "logspace" in spec:
        start, stop, num = spec["logspace"]
        vals = [float(v) for v in np.logspace(start, stop, int(num))]
        return trial.suggest_categorical(name, vals)
    raise ValueError(f"Cannot map param spec for '{name}' to Optuna: {spec}")


def run_optuna_sweep(
    name: str,
    seeds: list[int],
    parameters: dict[str, dict[str, Any]],
    runner: SweepableRunner,
    n_trials: int = 50,
    objective_metric: str = "safety_violation_rate",
    direction: str = "minimize",
    *,
    falsifier: bool = False,
    falsifier_metric: str = "safety_violations",
    safety_constraint: "tuple[str, float] | None" = None,
    viz: SweepViz | None = None,
    batch_runner: BatchSweepableRunner | None = None,
    # Legacy aliases
    skip_on_failure: bool | None = None,
    failure_metric: str | None = None,
) -> SweepRun:
    """Run an Optuna-driven parameter sweep.

    Each Optuna trial evaluates the scenario across all seeds and optimises
    the aggregated *objective_metric*.

    Parameters
    ----------
    falsifier : bool
        When *True*, stop iterating seeds on first failure and move to
        the next trial.
    falsifier_metric : str
        The result key checked for failure (default ``"safety_violations"``).
    safety_constraint : tuple[str, float] | None
        Optional ``(metric, max_value)`` pair.  When set, any trial whose
        aggregated *metric* exceeds *max_value* is penalised with ``inf``
        so that Optuna treats it as infeasible.
    viz : SweepViz | None
        Optional live visualization.

    Requires ``pip install cbfkit[optuna]``.
    """
    falsifier, falsifier_metric = _resolve_falsifier_kwargs(
        falsifier, falsifier_metric, skip_on_failure, failure_metric,
    )

    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for method='optuna'. "
            "Install it with: pip install cbfkit[optuna]"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    records: list[dict[str, Any]] = []
    param_combos: list[dict[str, Any]] = []
    per_combo_summaries: list[dict[str, Any]] = []

    progress = _make_progress()
    live_instance: Live | None = None
    trial_task: int | None = None
    seed_task: int | None = None

    def _build_live_renderable():
        if viz is not None:
            return Group(viz.render_header(), progress, viz.render())
        return progress

    def objective(trial) -> float:
        combo = {pname: _suggest_param(trial, pname, pspec)
                 for pname, pspec in parameters.items()}
        param_combos.append(combo)

        _process_combo(
            runner, seeds, combo, trial.number, records, per_combo_summaries,
            falsifier=falsifier, falsifier_metric=falsifier_metric,
            progress=progress, seed_task_id=seed_task, combo_task_id=trial_task,
            batch_runner=batch_runner, viz=None, live=None,
        )

        summary = per_combo_summaries[-1]
        obj_val = summary.get(objective_metric, 0.0)
        if safety_constraint is not None:
            sc_metric, sc_max = safety_constraint
            if summary.get(sc_metric, 0.0) > sc_max:
                obj_val = float("inf")

        if viz is not None:
            summary_with_obj = dict(summary)
            summary_with_obj[objective_metric] = obj_val
            viz.add_result(combo, summary_with_obj)
            live_instance.update(_build_live_renderable())

        return obj_val

    study = optuna.create_study(direction=direction)

    with Live(_build_live_renderable(), console=_console, refresh_per_second=4,
              transient=True, vertical_overflow="visible") as live, \
            _quiet_stdout():
        live_instance = live
        trial_task = progress.add_task("Trials", total=n_trials)
        seed_task = progress.add_task("  Seeds", total=len(seeds))
        study.optimize(objective, n_trials=n_trials)

    # Print final viz so it persists after Live exits
    if viz is not None:
        _console.print(viz.render_final())

    return SweepRun(
        scenario=name,
        seeds=seeds,
        param_combos=param_combos,
        records=records,
        per_combo_summaries=per_combo_summaries,
    )


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Fallback serializer for numpy types and other non-JSON objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_sweep_artifacts(run: SweepRun, output_dir: str | Path) -> None:
    """Write sweep results to JSON and CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "scenario": run.scenario,
        "seeds": run.seeds,
        "param_combos": run.param_combos,
        "per_combo_summaries": run.per_combo_summaries,
        "records": run.records,
    }
    (out / "sweep_results.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )

    if run.records:
        keys = sorted({k for rec in run.records for k in rec.keys()})
        with (out / "sweep_records.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(run.records)

    if run.per_combo_summaries:
        keys = sorted({k for s in run.per_combo_summaries for k in s.keys()})
        with (out / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(run.per_combo_summaries)
