"""Parameter sweep engine for benchmark scenarios."""

from __future__ import annotations

import csv
import itertools
import json
import random as pyrandom
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

import numpy as np
from tqdm import tqdm

from .metrics import summarize
from .registry import SweepableRunner


# ---------------------------------------------------------------------------
# Parameter space helpers
# ---------------------------------------------------------------------------


def expand_param_spec(spec: Dict[str, Any]) -> List[Any]:
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


def build_param_grid(parameters: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of all parameter value lists."""
    names = list(parameters.keys())
    value_lists = [expand_param_spec(parameters[n]) for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def sample_param_combos(
    parameters: Dict[str, Dict[str, Any]],
    n_samples: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
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
    seeds: List[int]
    param_combos: List[Dict[str, Any]]
    records: List[Dict[str, Any]]
    per_combo_summaries: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------------


def run_sweep(
    name: str,
    seeds: List[int],
    param_combos: List[Dict[str, Any]],
    runner: SweepableRunner,
) -> SweepRun:
    """Execute a scenario across all (seed, param_combo) pairs."""
    records: List[Dict[str, Any]] = []
    per_combo_summaries: List[Dict[str, Any]] = []

    total = len(param_combos) * len(seeds)
    with tqdm(total=total, desc=f"Sweep ({name})", unit="run") as pbar:
        for combo_idx, combo in enumerate(param_combos):
            combo_records: List[Dict[str, Any]] = []
            for seed in seeds:
                result = dict(runner(seed, combo))
                result["seed"] = seed
                result["combo_idx"] = combo_idx
                for k, v in combo.items():
                    result[f"param_{k}"] = v
                records.append(result)
                combo_records.append(result)
                pbar.update(1)

            summary = summarize(combo_records)
            summary["combo_idx"] = combo_idx
            for k, v in combo.items():
                summary[f"param_{k}"] = v
            per_combo_summaries.append(summary)

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
