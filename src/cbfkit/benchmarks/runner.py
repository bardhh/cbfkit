"""Benchmark runner and serialization utilities."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm

from .metrics import summarize
from .registry import registry


@dataclass(frozen=True)
class BenchmarkRun:
    scenario: str
    seeds: Sequence[int]
    records: list[dict[str, float | int | bool | str]]
    summary: dict[str, float]


def _parse_seeds(seeds: str | Iterable[int]) -> list[int]:
    if isinstance(seeds, str):
        if ":" in seeds:
            start, end = seeds.split(":", maxsplit=1)
            return list(range(int(start), int(end) + 1))
        return [int(token) for token in seeds.split(",") if token]
    return [int(seed) for seed in seeds]


def run_scenario(name: str, seeds: str | Iterable[int]) -> BenchmarkRun:
    parsed_seeds = _parse_seeds(seeds)
    spec = registry.scenario(name)

    records: list[dict[str, float | int | bool | str]] = []
    for seed in tqdm(parsed_seeds, desc=f"Seeds ({name})", unit="seed"):
        result = dict(spec.runner(seed))
        result["seed"] = seed
        records.append(result)

    return BenchmarkRun(
        scenario=name,
        seeds=parsed_seeds,
        records=records,
        summary=summarize(records),
    )


def write_artifacts(run: BenchmarkRun, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "scenario": run.scenario,
        "seeds": list(run.seeds),
        "summary": run.summary,
        "records": run.records,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    keys = sorted({key for rec in run.records for key in rec.keys()})
    with (out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(run.records)


def compare_runs(
    left: BenchmarkRun,
    right: BenchmarkRun,
    metric: str,
) -> dict[str, float]:
    left_value = float(left.summary.get(metric, 0.0))
    right_value = float(right.summary.get(metric, 0.0))
    return {
        "left": left_value,
        "right": right_value,
        "delta": right_value - left_value,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cbfkit benchmark scenario.")
    parser.add_argument("--scenario", required=True, help="Name of the scenario to run.")
    parser.add_argument("--seeds", default="0:4", help="Seeds range (start:end) or list (1,2,3).")
    parser.add_argument("--output", default="./benchmark_results", help="Directory for artifacts.")
    args = parser.parse_args()

    # Import locally to ensure registry is populated if running as script
    # (Though usually run as module `python -m cbfkit.benchmarks.runner` which imports package)
    # The __init__.py imports everything, so if run as module, registry is populated.

    try:
        print(f"Running scenario '{args.scenario}' with seeds {args.seeds}...")
        run = run_scenario(args.scenario, args.seeds)
        write_artifacts(run, args.output)

        print("\nSummary:")
        for k, v in run.summary.items():
            print(f"  {k}: {v}")
        print(f"\nArtifacts written to {args.output}")
    except KeyError as e:
        print(f"Error: {e}")
        # Print available scenarios
        print(f"Available scenarios: {', '.join(registry.names())}")
        exit(1)
