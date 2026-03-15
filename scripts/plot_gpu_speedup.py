#!/usr/bin/env python3
"""Plot CPU vs GPU speedup curves from Monte Carlo benchmark results.

Usage
-----
    # Run benchmarks on CPU and GPU first:
    JAX_PLATFORM_NAME=cpu cbfkit-bench run monte_carlo_gpu_speedup --seeds 0:2 --out results_cpu
    cbfkit-bench run monte_carlo_gpu_speedup --seeds 0:2 --out results_gpu

    # Then plot:
    python scripts/plot_gpu_speedup.py --cpu-dir results_cpu --gpu-dir results_gpu -o gpu_speedup.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_times(results_dir: Path) -> dict[int, list[float]]:
    """Load per-trial-count times from results.json + records.csv."""
    results_path = results_dir / "results.json"
    if not results_path.exists():
        sys.exit(f"Error: {results_path} not found")

    with open(results_path) as f:
        data = json.load(f)

    records = data.get("records", [data.get("summary", {})])
    times: dict[int, list[float]] = {}
    for rec in records:
        for key, val in rec.items():
            if key.startswith("time_"):
                n = int(key.split("_")[1])
                times.setdefault(n, []).append(float(val))
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CPU vs GPU Monte Carlo speedup")
    parser.add_argument("--cpu-dir", type=Path, required=True, help="CPU benchmark results dir")
    parser.add_argument("--gpu-dir", type=Path, required=True, help="GPU benchmark results dir")
    parser.add_argument("-o", "--output", type=Path, default=Path("gpu_speedup.png"))
    args = parser.parse_args()

    cpu_times = _load_times(args.cpu_dir)
    gpu_times = _load_times(args.gpu_dir)

    ns = sorted(set(cpu_times.keys()) & set(gpu_times.keys()))
    if not ns:
        sys.exit("No matching trial counts found between CPU and GPU results")

    cpu_means = [np.mean(cpu_times[n]) for n in ns]
    gpu_means = [np.mean(gpu_times[n]) for n in ns]
    speedups = [c / g for c, g in zip(cpu_means, gpu_means)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Wall-clock time comparison
    ax1.plot(ns, cpu_means, "o-", label="CPU", color="#2196F3", linewidth=2, markersize=8)
    ax1.plot(ns, gpu_means, "s-", label="GPU", color="#76B900", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Trials", fontsize=12)
    ax1.set_ylabel("Wall-Clock Time (s)", fontsize=12)
    ax1.set_title("Monte Carlo Simulation Time", fontsize=14)
    ax1.set_yscale("log")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Speedup ratio
    ax2.bar(range(len(ns)), speedups, tick_label=[str(n) for n in ns], color="#76B900", alpha=0.85)
    ax2.set_xlabel("Number of Trials", fontsize=12)
    ax2.set_ylabel("Speedup (CPU / GPU)", fontsize=12)
    ax2.set_title("GPU Speedup Factor", fontsize=14)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, axis="y", alpha=0.3)

    for i, s in enumerate(speedups):
        ax2.text(i, s + 0.1, f"{s:.1f}x", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("CBFKit: GPU-Accelerated Monte Carlo Safety Verification", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
