"""Smoke-run the MuJoCo examples in test mode (short horizon, no plots)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [
    "examples/mujoco/cart_pole_swingup.py",
    ("examples/mujoco/g1_scramble.py", "--planner", "mppi", "--proxy"),
    ("examples/mujoco/g1_scramble_social_eval.py", "--configs", "mppi", "--seeds", "0"),
    "examples/mujoco/g1_standup.py",
    "examples/mujoco/g1_navigate.py",
    "examples/mujoco/g1_plaza.py",
    "examples/mujoco/g1_model_distance.py",
    "examples/mujoco/g1_scramble.py",
]


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda s: s[0] if isinstance(s, tuple) else s)
def test_example_runs_in_test_mode(script):
    script, *args = script if isinstance(script, tuple) else (script,)
    env = dict(os.environ, CBFKIT_TEST_MODE="1", MPLBACKEND="Agg", JAX_PLATFORM_NAME="cpu")
    proc = subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert any(
        k in proc.stdout
        for k in (
            "mean upright-distance",
            "torso height",
            "h(x) min",
            "Disturbance bound",
            "| config |",
        )
    )
