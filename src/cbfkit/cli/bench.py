"""CLI for cbfkit-bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbfkit.benchmarks import compare_runs, registry, run_scenario, write_artifacts


def _cmd_list() -> int:
    for name in registry.names():
        spec = registry.scenario(name)
        print(f"{name}: {spec.description}")
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

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
