"""Metrics for benchmark outputs."""

from __future__ import annotations

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
    """Produce common summary statistics for benchmark outputs."""
    return {
        "num_runs": float(len(records)),
        "success_rate": rate(records, "success"),
        "safety_violation_rate": rate(records, "safety_violations"),
        "solver_failure_rate": rate(records, "solver_failures"),
        "avg_step_ms": mean(records, "avg_step_ms"),
    }
