"""Metrics for benchmark outputs."""

from __future__ import annotations

__all__ = ["mean", "rate", "summarize"]

from typing import Mapping, Sequence

Record = Mapping[str, float | int | bool | str]


def _to_float(value: float | int | bool | str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Non-numeric metric value: {value!r}")


def mean(records: Sequence[Record], key: str) -> float:
    if not records:
        return 0.0
    return sum(_to_float(record.get(key, 0.0)) for record in records) / len(records)


def rate(records: Sequence[Record], key: str) -> float:
    """Alias for mean over binary or count-like key values."""
    return mean(records, key)


def summarize(records: Sequence[Record]) -> dict[str, float]:
    """Produce common summary statistics for benchmark outputs.

    In addition to the fixed metrics, any extra numeric keys found in the
    records are averaged automatically so that scenario-specific metrics
    (e.g. ``final_goal_distance``, ``time_to_goal``) are carried through
    to the sweep summary without requiring changes here.
    """
    fixed: dict[str, float] = {
        "num_runs": float(len(records)),
        "success_rate": rate(records, "success"),
        "safety_violation_rate": rate(records, "safety_violations"),
        "solver_failure_rate": rate(records, "solver_failures"),
        "avg_step_ms": mean(records, "avg_step_ms"),
    }

    # Collect extra numeric keys from the first record (all records share the same schema)
    fixed_keys = set(fixed.keys()) | {"success", "safety_violations", "solver_failures"}
    extra_keys: set[str] = set()
    if records:
        for k, v in records[0].items():
            if k not in fixed_keys and isinstance(v, (int, float, bool)):
                extra_keys.add(k)

    for k in sorted(extra_keys):
        fixed[k] = mean(records, k)

    return fixed
