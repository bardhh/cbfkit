"""Plotting utilities for parameter sweep results."""

from __future__ import annotations

__all__ = ["plot_sweep"]

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_summaries(results_path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(results_path).read_text())
    return data["per_combo_summaries"]


def plot_sweep(
    results_path: str | Path,
    x_param: str,
    y_metric: str,
    *,
    hue_param: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    kind: str = "line",
) -> None:
    """Generate a sweep visualization from ``sweep_results.json``.

    Parameters
    ----------
    results_path : path to sweep_results.json
    x_param : parameter name for x-axis (without ``param_`` prefix)
    y_metric : metric name for y-axis
    hue_param : optional second parameter for colour grouping
    output_path : save figure to this path (shows interactively if *None*)
    kind : ``"line"``, ``"heatmap"``, or ``"pareto"``
    """
    import matplotlib.pyplot as plt

    summaries = _load_summaries(results_path)
    x_key = f"param_{x_param}"

    fig, ax = plt.subplots(figsize=(8, 5))

    if kind == "heatmap" and hue_param:
        _plot_heatmap(ax, summaries, x_key, f"param_{hue_param}", y_metric)
    elif kind == "pareto":
        _plot_pareto(ax, summaries, x_key, y_metric)
    elif hue_param:
        _plot_grouped_line(ax, summaries, x_key, y_metric, f"param_{hue_param}")
    else:
        x_vals = [s[x_key] for s in summaries]
        y_vals = [s[y_metric] for s in summaries]
        ax.plot(x_vals, y_vals, "o-")
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_metric)

    ax.set_title(f"{y_metric} vs {x_param}")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plot_grouped_line(ax, summaries, x_key, y_metric, hue_key) -> None:
    groups: Dict[Any, tuple[list, list]] = {}
    for s in summaries:
        h = s[hue_key]
        groups.setdefault(h, ([], []))
        groups[h][0].append(s[x_key])
        groups[h][1].append(s[y_metric])

    for label, (xs, ys) in sorted(groups.items()):
        ax.plot(xs, ys, "o-", label=f"{hue_key.removeprefix('param_')}={label}")
    ax.set_xlabel(x_key.removeprefix("param_"))
    ax.set_ylabel(y_metric)
    ax.legend()


def _plot_heatmap(ax, summaries, x_key, hue_key, y_metric) -> None:
    import numpy as np

    x_set = sorted(set(s[x_key] for s in summaries))
    h_set = sorted(set(s[hue_key] for s in summaries))
    grid = np.full((len(h_set), len(x_set)), np.nan)

    x_idx = {v: i for i, v in enumerate(x_set)}
    h_idx = {v: i for i, v in enumerate(h_set)}
    for s in summaries:
        grid[h_idx[s[hue_key]], x_idx[s[x_key]]] = s[y_metric]

    im = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_xticks(range(len(x_set)), [f"{v:.3g}" for v in x_set])
    ax.set_yticks(range(len(h_set)), [f"{v:.3g}" for v in h_set])
    ax.set_xlabel(x_key.removeprefix("param_"))
    ax.set_ylabel(hue_key.removeprefix("param_"))
    ax.figure.colorbar(im, ax=ax, label=y_metric)


def _plot_pareto(ax, summaries, x_key, y_metric) -> None:
    xs = [s[x_key] for s in summaries]
    ys = [s[y_metric] for s in summaries]
    ax.scatter(xs, ys, zorder=3)

    # Draw Pareto front (lower-left is better)
    pts = sorted(zip(xs, ys))
    front_x, front_y = [pts[0][0]], [pts[0][1]]
    best_y = pts[0][1]
    for x, y in pts[1:]:
        if y <= best_y:
            front_x.append(x)
            front_y.append(y)
            best_y = y
    ax.plot(front_x, front_y, "r--", linewidth=2, label="Pareto front")

    ax.set_xlabel(x_key.removeprefix("param_"))
    ax.set_ylabel(y_metric)
    ax.legend()
