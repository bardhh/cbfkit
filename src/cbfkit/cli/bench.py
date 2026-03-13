"""CLI for cbfkit-bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbfkit.benchmarks import compare_runs, registry, run_scenario, write_artifacts


def _cmd_list() -> int:
    for name in registry.names():
        spec = registry.scenario(name)
        params = ""
        if spec.sweepable_params:
            params = f"  [sweepable: {', '.join(spec.sweepable_params)}]"
        print(f"{name}: {spec.description}{params}")
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

    config = load_sweep_config(args.config)

    spec = registry.scenario(config.scenario)

    if spec.sweep_runner is not None:
        runner = spec.sweep_runner
    else:
        def runner(seed, params):
            return spec.runner(seed)

    if config.method == "optuna":
        print(
            f"Optuna sweep: {config.scenario} | "
            f"{config.n_samples} trials x {len(config.seeds)} seeds | "
            f"objective: {config.direction} {config.objective}"
        )
        result = run_optuna_sweep(
            config.scenario,
            config.seeds,
            config.parameters,
            runner,
            n_trials=config.n_samples,
            objective_metric=config.objective,
            direction=config.direction,
        )
    else:
        param_combos = resolve_param_combos(config)
        print(
            f"Sweep: {config.scenario} | "
            f"{len(param_combos)} combos x {len(config.seeds)} seeds = "
            f"{len(param_combos) * len(config.seeds)} runs"
        )
        result = run_sweep(config.scenario, config.seeds, param_combos, runner)

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
