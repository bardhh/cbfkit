"""Compare wall-time of JAXOPT (OSQP) / CVXOPT / Fast CBF-CLF QP solvers.

Each solver is the canonical entrypoint from ``cbfkit.optimization.quadratic_program``
(unified registry, PR #335). Sizes are chosen to match typical CBF-CLF-QP problems:
2-8 decision variables and 5-20 inequality constraints.

Usage::

    python benchmarks/qp_solver_comparison.py
    python benchmarks/qp_solver_comparison.py --out media/showcase/fast_qp_benchmark.png
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import numpy as np

from cbfkit.optimization.quadratic_program import get_solver

SOLVERS = ["jaxopt", "cvxopt", "fast"]
SIZES = [(2, 5), (4, 10), (8, 20)]


def build_problem(n_vars: int, n_constraints: int, seed: int = 0):
    """Build a feasible convex QP."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n_vars, n_vars))
    P = M.T @ M + np.eye(n_vars)
    q = rng.standard_normal(n_vars)
    G = rng.standard_normal((n_constraints, n_vars))
    h = 5.0 * np.ones(n_constraints)  # generous bounds => always feasible
    return jnp.asarray(P), jnp.asarray(q), jnp.asarray(G), jnp.asarray(h)


def time_solve(solver_name: str, args, n_reps: int = 50) -> float:
    """Return seconds per solve (mean over n_reps, with one warmup call)."""
    solver = get_solver(solver_name)
    P, q, G, h = args
    # Warmup (JIT / first-call setup)
    sol = solver(P, q, G, h)
    _ = float(sol.primal[0])
    t0 = time.perf_counter()
    for _ in range(n_reps):
        sol = solver(P, q, G, h)
        # Force evaluation for JAX solvers
        _ = float(sol.primal[0])
    return (time.perf_counter() - t0) / n_reps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="media/showcase/fast_qp_benchmark.png")
    parser.add_argument("--n_reps", type=int, default=50)
    args = parser.parse_args()

    timings_us = np.zeros((len(SIZES), len(SOLVERS)))
    print(f"\n{'size':>8s} | " + " | ".join(f"{s:>10s}" for s in SOLVERS))
    print("-" * (10 + 13 * len(SOLVERS)))
    for i, (n, m) in enumerate(SIZES):
        prob = build_problem(n, m)
        row = []
        for j, name in enumerate(SOLVERS):
            try:
                t = time_solve(name, prob, n_reps=args.n_reps)
            except Exception as e:
                print(f"   {name} failed at ({n},{m}): {e}")
                t = float("nan")
            timings_us[i, j] = t * 1e6
            row.append(f"{t * 1e6:>10.1f}")
        size_label = f"{n}x{m}"
        print(f"{size_label:>8s} | " + " | ".join(row) + "   (us/solve)")

    speedup = timings_us[:, :2] / timings_us[:, 2:3]  # vs fast
    print("\nSpeedup of 'fast' vs others:")
    for i, (n, m) in enumerate(SIZES):
        print(f"  {n}x{m}: jaxopt={speedup[i,0]:>7.1f}x, cvxopt={speedup[i,1]:>7.1f}x")

    # Plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(SIZES))
    w = 0.25
    label_map = {"jaxopt": "JAXOPT OSQP", "cvxopt": "CVXOPT", "fast": "CBFKit fast"}
    colors = {"jaxopt": "#888888", "cvxopt": "#d77777", "fast": "#3aa770"}
    for j, name in enumerate(SOLVERS):
        ax.bar(
            x + (j - 1) * w,
            timings_us[:, j],
            w,
            label=label_map[name],
            color=colors[name],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}x{m}" for n, m in SIZES])
    ax.set_xlabel("QP size (n_vars x n_constraints)")
    ax.set_ylabel("Wall time per solve (us, log)")
    ax.set_title("CBF-CLF QP solver wall-time - lower is better")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
