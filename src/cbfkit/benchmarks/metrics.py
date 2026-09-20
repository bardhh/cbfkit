"""Metrics for benchmark outputs."""

from __future__ import annotations

__all__ = ["mean", "rate", "summarize"]

import math
from typing import Mapping, Sequence

Record = Mapping[str, float | int | bool | str | None]


def _to_float(value: float | int | bool | str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
        raise ValueError(f"Non-finite metric value: {value!r}")
    raise ValueError(f"Non-numeric metric value: {value!r}")


def metric_value(record: Record, key: str) -> float:
    """Require a measured, finite metric for comparisons and decisions."""
    if key not in record:
        raise ValueError(f"Unknown or missing metric {key!r}.")
    value = record[key]
    if value is None:
        raise ValueError(f"Metric {key!r} is unavailable.")
    return _to_float(value)


def mean(records: Sequence[Record], key: str) -> float | None:
    """Average a metric; explicit None means the measurement is unavailable.

    Every record must declare the key. Partial measurements produce None,
    rather than averaging a selected subset or treating absent values as zero.
    """
    if not records:
        raise ValueError("Cannot summarize an empty benchmark run.")
    values = []
    for index, record in enumerate(records):
        if key not in record:
            raise ValueError(f"Record {index} is missing required metric {key!r}.")
        value = record[key]
        values.append(None if value is None else _to_float(value))
    if any(value is None for value in values):
        return None
    return math.fsum(value / len(values) for value in values if value is not None)


def rate(records: Sequence[Record], key: str) -> float | None:
    """Alias for mean over binary or count-like key values."""
    return mean(records, key)


def summarize(records: Sequence[Record]) -> dict[str, float | None]:
    """Produce common summary statistics for benchmark outputs.

    In addition to the fixed metrics, any extra numeric keys found in the
    records are averaged automatically so that scenario-specific metrics
    (e.g. ``final_goal_distance``, ``time_to_goal``) are carried through
    to the sweep summary without requiring changes here.
    """
    fixed: dict[str, float | None] = {
        "num_runs": float(len(records)),
        "success_rate": rate(records, "success"),
        "safety_violation_rate": rate(records, "safety_violations"),
        "solver_failure_rate": rate(records, "solver_failures"),
        "avg_step_ms": mean(records, "avg_step_ms"),
    }

    # Include optional metrics from every record. Missing optional measurements
    # remain unavailable in the summary instead of being silently discarded.
    fixed_keys = set(fixed.keys()) | {"success", "safety_violations", "solver_failures"}
    extra_keys: set[str] = set()
    for record in records:
        for k, v in record.items():
            if k not in fixed_keys and (v is None or isinstance(v, (int, float, bool))):
                extra_keys.add(k)

    for k in sorted(extra_keys):
        fixed[k] = mean([{k: record.get(k)} for record in records], k)

    return fixed
