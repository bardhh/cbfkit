"""Live CLI visualization for parameter sweeps using Rich."""

from __future__ import annotations

__all__ = ["SweepViz"]

import math
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _lerp_color(t: float) -> str:
    """Interpolate red (#ff0000) -> yellow (#ffff00) -> green (#00ff00) for t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        r = 255
        g = int(255 * (t / 0.5))
    else:
        r = int(255 * (1.0 - (t - 0.5) / 0.5))
        g = 255
    return f"#{r:02x}{g:02x}00"


def _color_for_value(
    val: float, lo: float, hi: float, direction: str,
) -> str:
    """Map *val* to a red-to-green color given the finite range [lo, hi]."""
    if not math.isfinite(val):
        return "#880000"
    if lo == hi:
        return _lerp_color(0.5)
    t = (val - lo) / (hi - lo)
    if direction == "minimize":
        t = 1.0 - t
    return _lerp_color(t)


def _render_grid(
    grid: dict[tuple[int, int], tuple[float, str, str]],
    w: int,
    h: int,
    title: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str = "",
    y_label: str = "",
    label_width: int = 7,
    direction: str = "minimize",
) -> Text:
    """Render a character-grid scatter plot as a Rich ``Text``."""
    lines: list = [f"  {title}"]
    pad = " " * label_width

    for row in range(h - 1, -1, -1):
        y_val = y_min + (y_max - y_min) * row / max(h - 1, 1)
        if row == h - 1 or row == 0 or row == h // 2:
            label = f"{y_val:>{label_width - 1}.2f} "
        else:
            label = pad
        line_text = Text(label, style="dim")
        line_text.append("\u2502", style="dim")
        for col in range(w):
            if (col, row) in grid:
                _, color, marker = grid[(col, row)]
                line_text.append(marker, style=color)
            else:
                line_text.append(" ")
        lines.append(line_text)

    lines.append(Text(pad + "\u2514" + "\u2500" * w, style="dim"))

    mid = x_min + (x_max - x_min) / 2
    x_nums = f"{pad} {x_min:<.2f}"
    gap = w // 2 - len(f"{x_min:<.2f}")
    x_nums += " " * max(0, gap) + f"{mid:.2f}"
    gap2 = w - (len(x_nums) - label_width - 1) - len(f"{x_max:.2f}")
    x_nums += " " * max(0, gap2) + f"{x_max:.2f}"
    lines.append(Text(x_nums, style="dim"))

    # Axis labels
    if x_label:
        x_label_line = Text(pad + " " + " " * ((w - len(x_label)) // 2) + x_label, style="dim italic")
        lines.append(x_label_line)

    # Color legend
    legend = Text("  ", style="dim")
    legend.append("\u25a0", style=_lerp_color(1.0))
    legend.append(" best ", style="dim")
    legend.append("\u25a0", style=_lerp_color(0.5))
    legend.append(" mid ", style="dim")
    legend.append("\u25a0", style=_lerp_color(0.0))
    legend.append(" worst", style="dim")
    lines.append(legend)

    result = Text()
    for i, line in enumerate(lines):
        if isinstance(line, str):
            result.append(line)
        else:
            result.append_text(line)
        if i < len(lines) - 1:
            result.append("\n")
    return result


class SweepViz:
    """Live-updating sweep visualization with table and scatter plot."""

    def __init__(
        self,
        objective_metric: str = "final_goal_distance",
        direction: str = "minimize",
        param_names: list[str] | None = None,
        *,
        sweep_name: str = "",
        mode: str = "grid",
        n_trials: int = 0,
        n_seeds: int = 0,
    ):
        self.objective_metric = objective_metric
        self.direction = direction
        self.param_names = param_names or []
        self.trials: list[dict[str, Any]] = []
        self._finite_lo: float = float("inf")
        self._finite_hi: float = float("-inf")
        self._best_idx: int = -1
        self._best_val: float = float("inf") if direction == "minimize" else float("-inf")
        self._safe_count: int = 0
        # Header metadata
        self.sweep_name = sweep_name
        self.mode = mode
        self.n_trials = n_trials
        self.n_seeds = n_seeds

    def add_result(self, combo: dict[str, Any], summary: dict[str, Any]) -> None:
        idx = len(self.trials)
        self.trials.append({"combo": dict(combo), "summary": dict(summary)})
        val = summary.get(self.objective_metric, 0.0)
        if summary.get("safety_violations", 0) == 0:
            self._safe_count += 1
        if not math.isfinite(val):
            return
        self._finite_lo = min(self._finite_lo, val)
        self._finite_hi = max(self._finite_hi, val)
        # Guard: if _best_val is non-finite (initial state), always accept
        if not math.isfinite(self._best_val):
            self._best_val = val
            self._best_idx = idx
        elif self.direction == "minimize" and val < self._best_val:
            self._best_val = val
            self._best_idx = idx
        elif self.direction != "minimize" and val > self._best_val:
            self._best_val = val
            self._best_idx = idx

    def render(self) -> Group:
        parts = []
        parts.append(self._render_best_line())
        scatter = self._render_scatter()
        if scatter:
            parts.append(scatter)
        parts.append(Text(""))
        parts.append(self._render_table())
        return Group(*parts)

    def render_final(self) -> Group:
        """Render the final post-sweep output with summary block."""
        parts = []
        scatter = self._render_scatter()
        if scatter:
            parts.append(scatter)
        parts.append(Text(""))
        parts.append(self._render_table())
        parts.append(Text(""))
        parts.append(self._render_summary())
        return Group(*parts)

    # ------------------------------------------------------------------
    # Header panel
    # ------------------------------------------------------------------

    def render_header(self) -> Panel:
        """Render a structured header panel."""
        mode_str = self.mode
        if self.mode == "optuna":
            mode_str = f"optuna ({self.direction} {self.objective_metric})"

        parts = []
        parts.append(f"[bold blue]Mode:[/bold blue] {mode_str}")
        parts.append(f"  [bold blue]Trials:[/bold blue] {self.n_trials}")
        parts.append(f"  [bold blue]Seeds:[/bold blue] {self.n_seeds}")
        content = "  ".join(parts)

        return Panel(
            content,
            title=f"[bold]{self.sweep_name}[/bold]",
            border_style="dim",
            expand=False,
            padding=(0, 1),
        )

    # ------------------------------------------------------------------
    # Best-so-far line
    # ------------------------------------------------------------------

    def _render_best_line(self) -> Text:
        if self._best_idx < 0 or not self.trials:
            return Text("")
        trial = self.trials[self._best_idx]
        combo = trial["combo"]
        val = self._best_val
        color = _color_for_value(val, self._finite_lo, self._finite_hi, self.direction)

        line = Text("  Best: ", style="bold")
        line.append(f"trial #{self._best_idx + 1}", style="bold cyan")
        line.append(" \u2192 ", style="dim")
        line.append(f"{self.objective_metric} = ", style="dim")
        line.append(f"{val:.4f}", style=f"bold {color}")
        params = ", ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in combo.items()
        )
        line.append(f"  ({params})", style="dim")
        line.append(f"    Safe: {self._safe_count}/{len(self.trials)}", style="dim")
        return line

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _render_table(self) -> Table:
        table = Table(
            title="Sweep Progress",
            title_style="bold cyan",
            border_style="dim",
            show_lines=False,
            expand=False,
            pad_edge=False,
            row_styles=["", "dim"],
        )
        table.add_column("#", style="dim", width=4, justify="right")
        for p in self.param_names:
            display = p if len(p) <= 12 else p[:11] + "\u2026"
            table.add_column(display, justify="right", min_width=8)
        table.add_column(self.objective_metric, justify="right", min_width=12)
        table.add_column("\u0394 best", justify="right", width=8)
        table.add_column("safe", justify="center", width=6)

        lo, hi = self._finite_lo, self._finite_hi
        for i, trial in enumerate(self.trials):
            combo = trial["combo"]
            summary = trial["summary"]
            obj_val = summary.get(self.objective_metric, 0.0)
            violations = summary.get("safety_violations", 0)
            safe = violations == 0
            color = _color_for_value(obj_val, lo, hi, self.direction)
            is_best = i == self._best_idx

            # Row number — star the best
            num_text = Text(f"\u2605 {i + 1}" if is_best else f"  {i + 1}")
            if is_best:
                num_text.stylize("bold yellow")

            row: list = [num_text]
            for p in self.param_names:
                v = combo.get(p, "")
                cell = f"{v:.3g}" if isinstance(v, float) else str(v)
                t = Text(cell)
                if is_best:
                    t.stylize("bold")
                row.append(t)

            # Objective value
            if math.isfinite(obj_val):
                obj_text = Text(f"{obj_val:.4f}", style=color)
                if is_best:
                    obj_text.stylize("bold")
                row.append(obj_text)
            else:
                row.append(Text("inf", style="#880000 bold"))

            # Delta from best
            if math.isfinite(obj_val) and self._best_idx >= 0 and math.isfinite(self._best_val):
                delta = obj_val - self._best_val
                if abs(delta) < 1e-9:
                    row.append(Text("--", style="dim"))
                else:
                    sign = "+" if delta > 0 else ""
                    row.append(Text(f"{sign}{delta:.3f}", style="dim"))
            else:
                row.append(Text("--", style="dim"))

            # Safety with violation count
            if safe:
                row.append(Text("\u2713", style="green"))
            else:
                count = int(violations) if isinstance(violations, (int, float)) else violations
                row.append(Text(f"\u2717 {count}", style="red bold"))

            table.add_row(*row)

        return table

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------

    def _render_scatter(self) -> Text | None:
        if len(self.trials) < 3:
            return None

        numeric_params = [
            p for p in self.param_names
            if any(isinstance(t["combo"].get(p), (int, float)) for t in self.trials)
        ]

        if self.mode == "optuna":
            return self._scatter_convergence()
        elif len(numeric_params) >= 2:
            return self._scatter_2d(numeric_params[0], numeric_params[1])
        elif len(numeric_params) == 1:
            return self._scatter_1d(numeric_params[0])
        return None

    def _populate_grid(
        self,
        coords: list[tuple[float, float, float]],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        w: int,
        h: int,
    ) -> dict[tuple[int, int], tuple[float, str, str]]:
        x_min, x_max = x_range
        y_min, y_max = y_range
        lo, hi = self._finite_lo, self._finite_hi
        grid: dict[tuple[int, int], tuple[float, str, str]] = {}

        for x, y, obj in coords:
            dx = (x_max - x_min) or 1.0
            dy = (y_max - y_min) or 1.0
            cx = int((x - x_min) / dx * (w - 1))
            cy = int((y - y_min) / dy * (h - 1))
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            color = _color_for_value(obj, lo, hi, self.direction)
            marker = "X" if not math.isfinite(obj) else "\u25cf"

            key = (cx, cy)
            if key not in grid:
                grid[key] = (obj, color, marker)
            else:
                prev_obj = grid[key][0]
                better = (obj < prev_obj) if self.direction == "minimize" else (obj > prev_obj)
                if better:
                    grid[key] = (obj, color, marker)

        return grid

    def _safe_range(self, lo: float, hi: float) -> tuple[float, float]:
        return (lo, hi + 1.0) if lo == hi else (lo, hi)

    def _scatter_2d(self, x_param: str, y_param: str) -> Text:
        W, H = 50, 15
        xs = [float(t["combo"][x_param]) for t in self.trials]
        ys = [float(t["combo"][y_param]) for t in self.trials]
        objs = [t["summary"].get(self.objective_metric, 0.0) for t in self.trials]

        x_range = self._safe_range(min(xs), max(xs))
        y_range = self._safe_range(min(ys), max(ys))

        grid = self._populate_grid(
            list(zip(xs, ys, objs)), x_range, y_range, W, H,
        )
        return _render_grid(
            grid, W, H, f"{y_param} vs {x_param}",
            x_range[0], x_range[1], y_range[0], y_range[1],
            x_label=x_param, y_label=y_param,
            direction=self.direction,
        )

    def _scatter_1d(self, x_param: str) -> Text:
        W, H = 50, 10
        xs = [float(t["combo"][x_param]) for t in self.trials]
        objs = [t["summary"].get(self.objective_metric, 0.0) for t in self.trials]

        if not math.isfinite(self._finite_lo):
            return Text("")

        x_range = self._safe_range(min(xs), max(xs))
        y_range = self._safe_range(self._finite_lo, self._finite_hi)

        coords = []
        for x, obj in zip(xs, objs):
            if math.isfinite(obj):
                coords.append((x, obj, obj))
            else:
                edge = y_range[1] if self.direction == "minimize" else y_range[0]
                coords.append((x, edge, obj))

        grid = self._populate_grid(coords, x_range, y_range, W, H)
        return _render_grid(
            grid, W, H, f"{self.objective_metric} vs {x_param}",
            x_range[0], x_range[1], y_range[0], y_range[1],
            x_label=x_param, y_label=self.objective_metric,
            label_width=9,
            direction=self.direction,
        )

    def _scatter_convergence(self) -> Text:
        """Render a convergence plot: trial # on x-axis, best-so-far on y-axis."""
        W, H = 50, 12
        objs = [t["summary"].get(self.objective_metric, 0.0) for t in self.trials]
        if not math.isfinite(self._finite_lo):
            return Text("")

        # Compute running best
        best_so_far = []
        curr_best = float("inf") if self.direction == "minimize" else float("-inf")
        for obj in objs:
            if math.isfinite(obj):
                if self.direction == "minimize":
                    curr_best = min(curr_best, obj)
                else:
                    curr_best = max(curr_best, obj)
            best_so_far.append(curr_best if math.isfinite(curr_best) else obj)

        x_range = self._safe_range(1.0, float(len(self.trials)))
        y_range = self._safe_range(self._finite_lo, self._finite_hi)
        lo, hi = self._finite_lo, self._finite_hi

        grid: dict[tuple[int, int], tuple[float, str, str]] = {}
        dx = (x_range[1] - x_range[0]) or 1.0
        dy = (y_range[1] - y_range[0]) or 1.0

        for i, (obj, best) in enumerate(zip(objs, best_so_far)):
            trial_num = float(i + 1)
            # Plot individual trial as dim dot
            if math.isfinite(obj):
                cx = int((trial_num - x_range[0]) / dx * (W - 1))
                cy = int((obj - y_range[0]) / dy * (H - 1))
                cx = max(0, min(W - 1, cx))
                cy = max(0, min(H - 1, cy))
                color = _color_for_value(obj, lo, hi, self.direction)
                key = (cx, cy)
                if key not in grid:
                    grid[key] = (obj, color, "\u25cb")

            # Plot best-so-far as solid dot (overwrites)
            if math.isfinite(best):
                cx = int((trial_num - x_range[0]) / dx * (W - 1))
                cy = int((best - y_range[0]) / dy * (H - 1))
                cx = max(0, min(W - 1, cx))
                cy = max(0, min(H - 1, cy))
                color = _color_for_value(best, lo, hi, self.direction)
                grid[(cx, cy)] = (best, color, "\u25cf")

        return _render_grid(
            grid, W, H,
            f"Convergence ({self.objective_metric})",
            x_range[0], x_range[1], y_range[0], y_range[1],
            x_label="trial #",
            y_label=self.objective_metric,
            label_width=9,
            direction=self.direction,
        )

    # ------------------------------------------------------------------
    # Post-sweep summary
    # ------------------------------------------------------------------

    def _render_summary(self) -> Panel:
        """Render a compact post-sweep summary block."""
        if not self.trials:
            return Panel("No trials completed.", border_style="dim")

        objs = [
            t["summary"].get(self.objective_metric, 0.0)
            for t in self.trials
        ]
        finite_objs = [o for o in objs if math.isfinite(o)]

        lines = []

        # Best trial
        if self._best_idx >= 0:
            best_trial = self.trials[self._best_idx]
            best_combo = best_trial["combo"]
            params = ", ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in best_combo.items()
            )
            color = _color_for_value(self._best_val, self._finite_lo, self._finite_hi, self.direction)
            line = Text("  Best:  ", style="bold")
            line.append(f"trial #{self._best_idx + 1}", style="bold cyan")
            line.append(" \u2192 ", style="dim")
            line.append(f"{self.objective_metric} = ", style="dim")
            line.append(f"{self._best_val:.4f}", style=f"bold {color}")
            line.append(f"  ({params})", style="dim")
            lines.append(line)

        # Worst + stats
        if finite_objs:
            if self.direction == "minimize":
                worst_val = max(finite_objs)
            else:
                worst_val = min(finite_objs)
            mean_val = sum(finite_objs) / len(finite_objs)
            if len(finite_objs) > 1:
                var = sum((x - mean_val) ** 2 for x in finite_objs) / (len(finite_objs) - 1)
                std_val = var ** 0.5
                stats_str = f"  Worst: {worst_val:.4f}  |  Mean: {mean_val:.4f} \u00b1 {std_val:.4f}"
            else:
                stats_str = f"  Worst: {worst_val:.4f}  |  Mean: {mean_val:.4f}"
            lines.append(Text(stats_str, style="dim"))

        # Safety and counts
        safe_line = f"  Safe: {self._safe_count}/{len(self.trials)} trials"
        inf_count = len(objs) - len(finite_objs)
        if inf_count > 0:
            safe_line += f"  |  Failed (inf): {inf_count}"
        lines.append(Text(safe_line, style="dim"))

        content = Text()
        for i, line in enumerate(lines):
            content.append_text(line)
            if i < len(lines) - 1:
                content.append("\n")

        return Panel(
            content,
            title="[bold]Sweep Complete[/bold]",
            border_style="green",
            expand=False,
            padding=(0, 1),
        )
