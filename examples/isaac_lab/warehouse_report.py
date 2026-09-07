"""Validate paired warehouse trials and summarize every recorded case."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(directory):
    results = [json.loads(p.read_text()) for p in sorted(directory.glob("seed-*/result.json"))]
    summary_path = directory / "summary.json"
    if not summary_path.exists() or not results:
        raise ValueError("Run did not finish: missing summary or results")
    expected = {r["name"] for r in json.loads(summary_path.read_text())}
    if {r["name"] for r in results} != expected:
        raise ValueError("Summary and result files disagree")
    for seed in sorted({r["seed"] for r in results}):
        pair = [r for r in results if r["seed"] == seed]
        for r in pair:
            for key in (
                "scenario_parameters",
                "initial_xy",
                "policy_path",
                "steps",
                "dt_s",
                "goal",
                "num_envs",
            ):
                if r[key] != pair[0][key]:
                    raise ValueError(f"Unpaired seed {seed}: {key}")
            n = r["num_envs"]
            if len(r["robot_link_names"]) != 17:
                raise ValueError("Unexpected contact coverage")
            if len(r["contact_channel_shapes"]) != 5 or any(
                s != [n, 4, 1, 17] for s in r["contact_channel_shapes"]
            ):
                raise ValueError(f"Unexpected contact history shape: {r['contact_channel_shapes']}")
            if r["filter_failures"] or r["command_residual_violations"]:
                raise ValueError(f"Failed or infeasible filtered commands: {r['name']}")
            if r["mode"] == "filtered" and r.get("filter_dtype") != "float64":
                raise ValueError(f"Float64 filtering was not recorded: {r['name']}")
            for key in (
                "collided",
                "fallen",
                "reached",
                "successful",
                "first_goal_s",
                "peak_obstacle_force_n",
            ):
                if len(r[key]) != n:
                    raise ValueError(f"Incomplete metrics: {r['name']}, {key}")
            collision = np.array(r["peak_obstacle_force_n"]) > r["contact_threshold_n"]
            if not np.array_equal(collision, r["collided"]):
                raise ValueError("Contact flags disagree with measured forces")
            success = np.array(r["reached"]) & ~collision & ~np.array(r["fallen"])
            if not np.array_equal(success, r["successful"]):
                raise ValueError("Success flags disagree with safety/task metrics")
    aggregates = {}
    for mode in sorted({r["mode"] for r in results}):
        group = [r for r in results if r["mode"] == mode]
        times = [t for r in group for t, ok in zip(r["first_goal_s"], r["successful"]) if ok]
        latency = [x for r in group for x in r["filter_ms_after_warmup"]]
        aggregates[mode] = {
            "cases": sum(r["num_envs"] for r in group),
            "contacts": sum(sum(r["collided"]) for r in group),
            "falls": sum(sum(r["fallen"]) for r in group),
            "goals": sum(sum(r["reached"]) for r in group),
            "successful": sum(sum(r["successful"]) for r in group),
            "median_successful_goal_s": float(np.median(times)) if times else None,
            "filter_ms_p50": float(np.percentile(latency, 50)) if latency else None,
            "filter_ms_p95": float(np.percentile(latency, 95)) if latency else None,
            "intervention_fraction": float(np.mean([r["intervention_fraction"] for r in group])),
            "min_command_residual": (
                min(r["min_command_residual"] for r in group) if mode == "filtered" else None
            ),
        }
    return {"seeds": sorted({r["seed"] for r in results}), "modes": aggregates}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(summarize(args.directory), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)
