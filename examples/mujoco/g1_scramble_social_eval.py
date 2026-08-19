"""Goal-seeking nominal vs social MPPI in the G1 scramble: the intrusiveness table.

Runs ``g1_scramble.run`` for each planner configuration over several seeds (crowd layouts)
and writes ``results/g1_scramble_social_mppi.{md,json}``. Configurations:

* ``goal``      -- P-law to the goal, soft CBF-QP does all the avoiding (the baseline);
* ``mppi-eff``  -- MPPI with *only* progress / collision / smoothness terms (no proxemics,
                   no time-to-collision, no slow-near-people, no pass side): what a
                   collision-aware but socially blind planner does;
* ``mppi``      -- the social MPPI (``g1_scramble.DEFAULT_WEIGHTS`` = the tuned
                   ``SocialCostWeights`` defaults + keep-left).

``--proxy`` (default here) uses the identified 2-D reduced model of the G1 + policy for fast
multi-seed statistics; ``--g1`` runs the MJX humanoid (minutes per run). Measured numbers
are in ``results/g1_scramble_social_mppi.md`` (regenerate with this script).

    python examples/mujoco/g1_scramble_social_eval.py [--seeds 0 1 2 3 4] [--g1] [--configs goal mppi]
"""

import argparse
import json
import os
import sys
from dataclasses import replace

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import numpy as np

import examples.mujoco.g1_scramble as sc  # noqa: E402

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = sc.RESULTS_DIR

CONFIGS = {
    "goal": dict(planner="goal", weights=sc.DEFAULT_WEIGHTS),
    "mppi-eff": dict(
        planner="mppi",
        weights=replace(sc.DEFAULT_WEIGHTS, proxemics=0.0, ttc=0.0, slow=0.0, pass_side=0.0),
    ),
    "mppi": dict(planner="mppi", weights=sc.DEFAULT_WEIGHTS),
}

COLUMNS = [
    ("crossed", "crossed", "{}"),
    ("crossing_time_s", "time [s]", "{:.1f}"),
    ("waiting_frac", "waiting", "{:.0%}"),
    ("path_ratio", "path ratio", "{:.2f}"),
    ("closest_m", "closest [m]", "{:.2f}"),
    ("h_min", "h_min", "{:+.2f}"),
    ("intimate_ped_s", "intimate [ped-s]", "{:.1f}"),
    ("personal_ped_s", "personal [ped-s]", "{:.1f}"),
    ("front_intrusion_ped_s", "front intrusion [ped-s]", "{:.1f}"),
    ("front_intrusions_n", "peds cut in front of", "{:.1f}"),
    ("intimate_rate", "intimate /10 s", "{:.2f}"),
    ("personal_rate", "personal /10 s", "{:.2f}"),
    ("front_rate", "front /10 s", "{:.2f}"),
    ("ped_deviation_mean_m", "crowd deviation [m]", "{:.2f}"),
    ("ped_slowdown_ped_s", "crowd slowdown [ped-s]", "{:.1f}"),
    ("robot_turn_rad", "robot turning [rad]", "{:.1f}"),
    ("robot_jerk_rms", "jerk rms", "{:.2f}"),
    ("cbf_active_frac", "CBF active", "{:.0%}"),
    ("slack_frac", "slack", "{:.1%}"),
]


def summarize(rows):
    """Mean over seeds of every numeric metric (``crossed`` -> fraction)."""
    out = {}
    keys = [c[0] for c in COLUMNS] + [
        "human_intimate_rate",
        "human_personal_rate",
        "human_front_rate",
    ]
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean([float(v) for v in vals])) if vals else float("nan")
    out["n"] = len(rows)
    return out


def to_markdown(summary, seeds, proxy):
    head = "| config | " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|---|" + "|".join("---:" for _ in COLUMNS) + "|"
    lines = [
        f"# Goal nominal vs social MPPI -- G1 scramble ({'2-D lagged proxy' if proxy else 'MJX G1'}, "
        f"{sc.N_PED} pedestrians, seeds {list(seeds)}, means over seeds)",
        "",
        head,
        sep,
    ]
    for name, s in summary.items():
        cells = []
        for key, _, fmt in COLUMNS:
            v = s[key]
            cells.append(fmt.format(v) if key != "crossed" else f"{v:.0%}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    first = next(iter(summary.values()))
    human = {
        "intimate_rate": first["human_intimate_rate"],
        "personal_rate": first["human_personal_rate"],
        "front_rate": first["human_front_rate"],
    }
    cells = [("{:.2f}".format(human[k]) if k in human else "--") for k, _, _ in COLUMNS]
    lines.append("| *a pedestrian (robot-free crowd)* | " + " | ".join(cells) + " |")
    lines += [
        "",
        "`/10 s` rates are pedestrian-seconds per 10 s the agent spent inside the intersection; the "
        "last row is the same statistic for a pedestrian among pedestrians (robot absent) -- the "
        "human norm for this crowd.",
        "intimate/personal: pedestrian-seconds with < 0.45 m / < 1.2 m between bodies; front intrusion: "
        "pedestrian-seconds with the robot inside the +-45 deg cone within 1.5 m ahead of a walking "
        "pedestrian; crowd deviation: mean over pedestrians of the max position difference to the "
        "robot-free rollout; crowd slowdown: robot-attributable speed loss (vs robot-free) within 2.5 m, "
        "in pedestrian-seconds; CBF active: steps where the QP changed the planner's acceleration.",
    ]
    return "\n".join(lines) + "\n"


def main(seeds=(0, 1, 2, 3, 4), proxy=True, configs=("goal", "mppi-eff", "mppi"), duration=None):
    duration = duration or (90.0 if proxy else sc.DEFAULT_DURATION)
    if TEST_MODE:
        duration = 2.0  # smoke run: enough steps to exercise the pipeline, short JIT
    per_seed = {c: [] for c in configs}
    for c in configs:
        for s in seeds:
            r = sc.run(duration, s, proxy=proxy, verbose=False, **CONFIGS[c])
            m = r["metrics"]
            per_seed[c].append(m)
            print(
                f"[{c} seed {s}] crossed={m['crossed']} t={m['crossing_time_s']:.1f}s "
                f"intimate={m['intimate_ped_s']:.1f} front={m['front_intrusion_ped_s']:.1f} "
                f"dev={m['ped_deviation_mean_m']:.2f} wait={m['waiting_frac']:.0%} "
                f"cbf={m['cbf_active_frac']:.0%} h_min={m['h_min']:+.2f} ({m['wall_s']:.0f}s)"
            )
    summary = {c: summarize(per_seed[c]) for c in configs}
    md = to_markdown(summary, seeds, proxy)
    print(md)
    if TEST_MODE:
        return summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "proxy" if proxy else "g1"
    with open(os.path.join(RESULTS_DIR, f"g1_scramble_social_mppi_{tag}.md"), "w") as f:
        f.write(md)
    with open(os.path.join(RESULTS_DIR, f"g1_scramble_social_mppi_{tag}.json"), "w") as f:
        json.dump({"summary": summary, "per_seed": per_seed, "seeds": list(seeds)}, f, indent=1)
    print(f"saved results/g1_scramble_social_mppi_{tag}.{{md,json}}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--g1", action="store_true", help="MJX G1 instead of the 2-D proxy")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--duration", type=float, default=None)
    a = p.parse_args()
    if TEST_MODE:
        a.seeds = a.seeds[:1]
    main(tuple(a.seeds), not a.g1, tuple(a.configs), a.duration)
