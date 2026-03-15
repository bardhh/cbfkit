"""CLI for cbfkit-bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbfkit.benchmarks import compare_runs, registry, run_scenario, write_artifacts


def _cmd_list() -> int:
    import yaml
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # -- Scenarios table --
    sc_table = Table(
        title="Scenarios",
        title_style="bold cyan",
        show_header=True,
        header_style="bold",
        border_style="dim",
        pad_edge=False,
        show_lines=True,
        expand=True,
    )
    sc_table.add_column("Scenario", style="green bold", no_wrap=True, min_width=24)
    sc_table.add_column("Description", ratio=2)
    sc_table.add_column("Sweepable Params", style="yellow", ratio=1)

    for name in registry.names():
        spec = registry.scenario(name)
        params = (
            ", ".join(spec.sweepable_params)
            if spec.sweepable_params
            else Text("--", style="dim")
        )
        sc_table.add_row(name, spec.description, params)

    console.print()
    console.print(sc_table)

    # -- Sweep configs table --
    configs_dir = Path("configs")
    config_files = sorted(configs_dir.rglob("*.yaml")) if configs_dir.is_dir() else []

    registered = set(registry.names())

    if config_files:
        cfg_table = Table(
            title="Sweep Configs",
            title_style="bold cyan",
            show_header=True,
            header_style="bold",
            border_style="dim",
            pad_edge=False,
            expand=True,
        )
        cfg_table.add_column("Run", no_wrap=True, ratio=3)
        cfg_table.add_column("Scenario", style="green", no_wrap=True, ratio=2)
        cfg_table.add_column("Method", style="blue", no_wrap=True, justify="center", min_width=8)
        cfg_table.add_column("Details", no_wrap=True, ratio=2)

        for cfg_path in config_files:
            try:
                raw = yaml.safe_load(cfg_path.read_text())
            except Exception:
                continue
            sweep = raw.get("sweep", {})
            scenario = raw.get("scenario", "?")
            method = sweep.get("method", "grid")

            # Scenario availability
            if scenario not in registered:
                scenario_cell = Text(f"{scenario} (not registered)", style="red")
            else:
                scenario_cell = scenario

            details_parts: list[str] = []
            if sweep.get("skip_on_failure"):
                details_parts.append("[yellow]skip on failure[/yellow]")
            n_params = len(sweep.get("parameters", {}))
            if method == "optuna":
                n = sweep.get("n_samples", "?")
                direction = sweep.get("direction", "min")[:3]
                obj = sweep.get("objective", "")
                details_parts.append(f"{n} trials {direction}. {obj}, {n_params} params")
            else:
                details_parts.append(f"{n_params} params")

            detail_str = ", ".join(details_parts) if details_parts else Text("--", style="dim")
            run_cmd = f"[green]cbfkit-bench sweep[/green] {cfg_path}"
            cfg_table.add_row(run_cmd, scenario_cell, method, detail_str)

        console.print()
        console.print(cfg_table)

    console.print(
        f"\n  [dim]{sc_table.row_count} scenario(s), "
        f"{len(config_files)} sweep config(s)[/dim]"
    )

    console.print(
        "\n  [bold]Quick start:[/bold]\n"
        "    [green]cbfkit-bench run[/green] <scenario>"
        "                    [dim]# run with default seeds 0:9[/dim]\n"
        "    [green]cbfkit-bench run[/green] <scenario> [yellow]--seeds 0:4[/yellow]"
        "     [dim]# custom seed range[/dim]\n"
        "    [green]cbfkit-bench sweep[/green] <config>"
        "                   [dim]# run a parameter sweep[/dim]\n"
        "    [green]cbfkit-bench compare[/green] <scenario>"
        "                [dim]# compare two seed ranges[/dim]\n"
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    run = run_scenario(args.scenario, args.seeds)
    if args.out:
        write_artifacts(run, args.out)
    print(json.dumps(run.summary, indent=2))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    left = run_scenario(args.scenario, args.left_seeds)
    right = run_scenario(args.scenario, args.right_seeds)
    delta = compare_runs(left, right, args.metric)
    print(json.dumps(delta, indent=2))
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from cbfkit.benchmarks.sweep import run_sweep, run_optuna_sweep, write_sweep_artifacts
    from cbfkit.benchmarks.sweep_config import load_sweep_config, resolve_param_combos
    from cbfkit.benchmarks.sweep_viz import SweepViz

    config = load_sweep_config(args.config)

    spec = registry.scenario(config.scenario)

    if spec.sweep_runner is not None:
        runner = spec.sweep_runner
    else:
        def runner(seed, params):
            return spec.runner(seed)

    batch_runner = spec.batch_sweep_runner

    # Build live viz unless disabled
    viz = None
    objective = config.objective if config.method == "optuna" else "violation_rate"
    direction = config.direction if config.method == "optuna" else "minimize"

    if config.method == "optuna":
        if not args.no_viz:
            viz = SweepViz(
                objective_metric=objective,
                direction=direction,
                param_names=list(config.parameters.keys()),
                sweep_name=config.scenario,
                mode="optuna",
                n_trials=config.n_samples,
                n_seeds=len(config.seeds),
            )
        sc = config.safety_constraint
        result = run_optuna_sweep(
            config.scenario,
            config.seeds,
            config.parameters,
            runner,
            n_trials=config.n_samples,
            objective_metric=config.objective,
            direction=config.direction,
            skip_on_failure=config.skip_on_failure,
            failure_metric=config.failure_metric,
            safety_constraint=(sc.metric, sc.max) if sc else None,
            viz=viz,
            batch_runner=batch_runner,
        )
    else:
        param_combos = resolve_param_combos(config)
        if not args.no_viz:
            viz = SweepViz(
                objective_metric=objective,
                direction=direction,
                param_names=list(config.parameters.keys()),
                sweep_name=config.scenario,
                mode="grid",
                n_trials=len(param_combos),
                n_seeds=len(config.seeds),
            )
        result = run_sweep(
            config.scenario, config.seeds, param_combos, runner,
            skip_on_failure=config.skip_on_failure,
            failure_metric=config.failure_metric,
            viz=viz,
            batch_runner=batch_runner,
        )

    write_sweep_artifacts(result, config.output_dir)
    print(f"Results written to {config.output_dir}")
    return 0


def _cmd_sweep_plot(args: argparse.Namespace) -> int:
    from cbfkit.benchmarks.sweep_plot import plot_sweep

    plot_sweep(
        args.results,
        x_param=args.x_param,
        y_metric=args.y_metric,
        hue_param=args.hue,
        output_path=args.output,
        kind=args.kind,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cbfkit-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")

    run_parser = sub.add_parser("run")
    run_parser.add_argument("scenario")
    run_parser.add_argument("--seeds", default="0:9", help="seed range a:b or csv list")
    run_parser.add_argument("--out", type=Path, default=None)

    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("scenario")
    cmp_parser.add_argument("--left-seeds", default="0:9")
    cmp_parser.add_argument("--right-seeds", default="10:19")
    cmp_parser.add_argument("--metric", default="safety_violation_rate")

    sweep_parser = sub.add_parser("sweep", help="Run a parameter sweep from YAML config")
    sweep_parser.add_argument("config", type=Path, help="Path to sweep YAML config file")
    sweep_parser.add_argument(
        "--no-viz", action="store_true", default=False,
        help="Disable live visualization (table + scatter plot)",
    )

    plot_parser = sub.add_parser("sweep-plot", help="Plot sweep results")
    plot_parser.add_argument("results", type=Path, help="Path to sweep_results.json")
    plot_parser.add_argument("--x-param", required=True, help="Parameter for x-axis")
    plot_parser.add_argument("--y-metric", required=True, help="Metric for y-axis")
    plot_parser.add_argument("--hue", default=None, help="Second parameter for grouping")
    plot_parser.add_argument(
        "--kind", default="line", choices=["line", "heatmap", "pareto"],
    )
    plot_parser.add_argument("--output", type=Path, default=None, help="Save figure to path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _cmd_list()
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "sweep":
        return _cmd_sweep(args)
    if args.command == "sweep-plot":
        return _cmd_sweep_plot(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
