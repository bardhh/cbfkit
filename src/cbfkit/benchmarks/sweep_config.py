"""YAML configuration loader for parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .runner import _parse_seeds
from .sweep import build_param_grid, sample_param_combos


@dataclass(frozen=True)
class SweepConfig:
    scenario: str
    seeds: List[int]
    method: str
    parameters: Dict[str, Dict[str, Any]]
    n_samples: int
    output_dir: str
    objective: str = "safety_violation_rate"
    direction: str = "minimize"


def load_sweep_config(path: str | Path) -> SweepConfig:
    """Parse a YAML sweep config file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    sweep_block = raw.get("sweep", {})
    output_block = raw.get("output", {})

    return SweepConfig(
        scenario=raw["scenario"],
        seeds=_parse_seeds(raw.get("seeds", "0:4")),
        method=sweep_block.get("method", "grid"),
        parameters=sweep_block.get("parameters", {}),
        n_samples=sweep_block.get("n_samples", 50),
        output_dir=output_block.get("dir", "./sweep_results"),
        objective=sweep_block.get("objective", "safety_violation_rate"),
        direction=sweep_block.get("direction", "minimize"),
    )


def resolve_param_combos(config: SweepConfig) -> List[Dict[str, Any]]:
    """Generate parameter combinations based on the config method.

    For ``method='optuna'``, returns an empty list since Optuna generates
    parameters dynamically.
    """
    if config.method == "optuna":
        return []
    if config.method == "random":
        return sample_param_combos(config.parameters, config.n_samples)
    return build_param_grid(config.parameters)
