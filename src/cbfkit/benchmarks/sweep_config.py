"""YAML configuration loader for parameter sweeps."""

from __future__ import annotations

__all__ = [
    "ObstacleItemSpec",
    "ObstaclesSpec",
    "SafetyConstraint",
    "SweepConfig",
    "SweepConfigError",
    "load_sweep_config",
    "resolve_param_combos",
]

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .runner import _parse_seeds
from .sweep import build_param_grid, sample_param_combos

_VALID_METHODS = {"grid", "random", "optuna"}
_VALID_DIRECTIONS = {"minimize", "maximize"}
_VALID_PARAM_SPEC_KEYS = {"values", "linspace", "logspace", "range", "int_range"}
_VALID_OBSTACLE_TYPES = {"circular", "ellipsoidal"}
_REQUIRED_OBSTACLE_PROPS = {
    "circular": {"center", "radius"},
    "ellipsoidal": {"center", "semi_axes"},
}


@dataclass(frozen=True)
class SafetyConstraint:
    """Hard constraint applied to Optuna trials.

    Trials where the *metric* exceeds *max* are penalized with ``inf``
    so that Optuna steers away from unsafe regions.
    """

    metric: str
    max: float


@dataclass(frozen=True)
class ObstacleItemSpec:
    """A single obstacle with fixed and/or sweepable properties."""

    fixed: Dict[str, Any]
    sweepable: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class ObstaclesSpec:
    """Obstacle layout specification from YAML config."""

    type: str  # "circular" or "ellipsoidal"
    items: List[ObstacleItemSpec]


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
    falsifier: bool = False
    falsifier_metric: str = "safety_violations"
    safety_constraint: SafetyConstraint | None = None
    obstacles: ObstaclesSpec | None = None


class SweepConfigError(ValueError):
    """Raised when a sweep config file is invalid."""


def _is_sweep_spec(value: Any) -> bool:
    """Check if a value is a sweep parameter spec (dict with a recognized key)."""
    return isinstance(value, dict) and bool(set(value.keys()) & _VALID_PARAM_SPEC_KEYS)


def _parse_obstacles(raw_obstacles: dict) -> ObstaclesSpec:
    """Parse the ``obstacles`` YAML block into an ``ObstaclesSpec``."""
    obs_type = raw_obstacles.get("type")
    if obs_type not in _VALID_OBSTACLE_TYPES:
        raise SweepConfigError(
            f"obstacles.type must be one of {sorted(_VALID_OBSTACLE_TYPES)}, "
            f"got: {obs_type!r}"
        )

    raw_items = raw_obstacles.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise SweepConfigError("obstacles.items must be a non-empty list.")

    required = _REQUIRED_OBSTACLE_PROPS[obs_type]
    items: List[ObstacleItemSpec] = []
    for i, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise SweepConfigError(f"obstacles.items[{i}] must be a mapping.")
        missing = required - set(raw_item.keys())
        if missing:
            raise SweepConfigError(
                f"obstacles.items[{i}] is missing required keys: {sorted(missing)}"
            )
        fixed: Dict[str, Any] = {}
        sweepable: Dict[str, Dict[str, Any]] = {}
        for prop, value in raw_item.items():
            if _is_sweep_spec(value):
                sweepable[prop] = value
            else:
                fixed[prop] = value
        items.append(ObstacleItemSpec(fixed=fixed, sweepable=sweepable))

    return ObstaclesSpec(type=obs_type, items=items)


def _extract_obstacle_sweep_params(
    obs_spec: ObstaclesSpec,
) -> Dict[str, Dict[str, Any]]:
    """Generate synthetic sweep parameter entries from sweepable obstacle properties.

    For example, obstacle item 0 with ``radius: {linspace: [0.3, 1.5, 5]}``
    produces ``{"obstacle_0_radius": {"linspace": [0.3, 1.5, 5]}}``.
    """
    params: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(obs_spec.items):
        for prop, spec in item.sweepable.items():
            params[f"obstacle_{i}_{prop}"] = spec
    return params


def _build_obstacle_fixed_params(obs_spec: ObstaclesSpec) -> Dict[str, Any]:
    """Build the dict of fixed obstacle metadata to inject into runner params.

    Keys use a leading underscore (e.g. ``_obstacle_count``) to distinguish
    them from sweep parameters.
    """
    fixed: Dict[str, Any] = {
        "_obstacle_count": len(obs_spec.items),
        "_obstacle_type": obs_spec.type,
    }
    for i, item in enumerate(obs_spec.items):
        for prop, value in item.fixed.items():
            fixed[f"_obstacle_{i}_{prop}"] = value
    return fixed


def _validate_config(raw: dict) -> None:
    """Validate raw YAML structure and raise ``SweepConfigError`` on problems."""
    if "scenario" not in raw:
        raise SweepConfigError("Config must specify a 'scenario' field.")

    sweep_block = raw.get("sweep", {})

    method = sweep_block.get("method", "grid")
    if method not in _VALID_METHODS:
        raise SweepConfigError(
            f"Unknown sweep method '{method}'. Must be one of: {sorted(_VALID_METHODS)}"
        )

    direction = sweep_block.get("direction", "minimize")
    if direction not in _VALID_DIRECTIONS:
        raise SweepConfigError(
            f"Unknown direction '{direction}'. Must be one of: {sorted(_VALID_DIRECTIONS)}"
        )

    parameters = sweep_block.get("parameters", {})
    for pname, pspec in parameters.items():
        if not isinstance(pspec, dict):
            raise SweepConfigError(
                f"Parameter '{pname}' must be a mapping (got {type(pname).__name__})."
            )
        spec_keys = set(pspec.keys()) - {"log"}  # 'log' is a modifier for 'range'
        if not spec_keys & _VALID_PARAM_SPEC_KEYS:
            raise SweepConfigError(
                f"Parameter '{pname}' has no recognized spec key. "
                f"Expected one of: {sorted(_VALID_PARAM_SPEC_KEYS)}, got: {sorted(pspec.keys())}"
            )

    sc_raw = sweep_block.get("safety_constraint")
    if sc_raw is not None:
        if not isinstance(sc_raw, dict) or "metric" not in sc_raw or "max" not in sc_raw:
            raise SweepConfigError(
                "safety_constraint must be a mapping with 'metric' and 'max' keys."
            )


def load_sweep_config(path: str | Path) -> SweepConfig:
    """Parse a YAML sweep config file.

    Raises ``SweepConfigError`` for invalid configs.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    _validate_config(raw)

    sweep_block = raw.get("sweep", {})
    output_block = raw.get("output", {})

    sc_raw = sweep_block.get("safety_constraint")
    safety_constraint = (
        SafetyConstraint(metric=sc_raw["metric"], max=float(sc_raw["max"]))
        if sc_raw
        else None
    )

    # Accept both canonical names (falsifier/falsifier_metric) and legacy
    # aliases (skip_on_failure/failure_metric) for backward compatibility.
    falsifier = sweep_block.get(
        "falsifier", sweep_block.get("skip_on_failure", False)
    )
    falsifier_metric = sweep_block.get(
        "falsifier_metric", sweep_block.get("failure_metric", "safety_violations")
    )

    # Parse obstacle config and merge synthetic sweep params
    obstacles: ObstaclesSpec | None = None
    parameters = dict(sweep_block.get("parameters", {}))
    if "obstacles" in raw:
        obstacles = _parse_obstacles(raw["obstacles"])
        parameters.update(_extract_obstacle_sweep_params(obstacles))

    return SweepConfig(
        scenario=raw["scenario"],
        seeds=_parse_seeds(raw.get("seeds", "0:4")),
        method=sweep_block.get("method", "grid"),
        parameters=parameters,
        n_samples=sweep_block.get("n_samples", 50),
        output_dir=output_block.get("dir", "./sweep_results"),
        objective=sweep_block.get("objective", "safety_violation_rate"),
        direction=sweep_block.get("direction", "minimize"),
        falsifier=falsifier,
        falsifier_metric=falsifier_metric,
        safety_constraint=safety_constraint,
        obstacles=obstacles,
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
