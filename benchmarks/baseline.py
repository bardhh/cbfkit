"""Repeat batched-filter measurements in fresh CPU processes and save the raw results.

Run with a locked environment: python benchmarks/baseline.py --output results/baseline.json
"""

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 1024])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min([args.steps, args.repeats, *args.batches]) <= 0:
        parser.error("batches, steps and repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        JAX_PLATFORM_NAME="cpu",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_COMPILATION_CACHE="false",
        CBFKIT_QP_SOLVER="fast",
    )
    rows = []
    for batch in args.batches:
        for repeat in range(args.repeats):
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "benchmarks/batched_safety_filter.py"),
                    "--batches",
                    str(batch),
                    "--steps",
                    str(args.steps),
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                timeout=300,
            )
            if result.returncode:
                raise RuntimeError(result.stderr[-4000:])
            row = json.loads(result.stdout)
            row["repeat"] = repeat
            rows.append(row)
            print(f"batch={batch} repeat={repeat + 1}: {row['mean_batch_ms']:.3f} ms", flush=True)
            if row["constraint_violations"] or row["failed_controls"]:
                raise RuntimeError("Baseline produced unsafe or failed controls")
    summaries = []
    metrics = [
        "warmup_s",
        "warmup_backend_compile_s",
        "mean_batch_ms",
        "p95_batch_ms",
        "controls_per_second",
        "peak_process_rss_bytes",
    ]
    for batch in args.batches:
        group = [row for row in rows if row["batch"] == batch]
        summary = {"batch": batch}
        for metric in metrics:
            values = [row[metric] for row in group if row[metric] is not None]
            summary[metric] = (
                {"median": statistics.median(values), "min": min(values), "max": max(values)}
                if values
                else None
            )
        summaries.append(summary)
    report = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "persistent_compilation_cache": False,
        "memory_scope": "process high-water host RSS, including imports; not device memory",
        "compile_scope": "JAX backend compile events during two warmup calls; excludes tracing",
        "results": rows,
        "summary": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
