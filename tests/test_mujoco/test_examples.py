"""Smoke-run the MuJoCo examples in test mode (short horizon, no plots)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
G1 = pytest.mark.g1_mjx  # every G1 script compiles the MJX humanoid step
EXAMPLES = [
    "examples/mujoco/cart_pole_swingup.py",
    pytest.param(("examples/mujoco/g1_scramble.py", "--planner", "mppi", "--proxy"), marks=G1),
    pytest.param(
        (
            "examples/mujoco/g1_scramble.py",
            "--footprint",
            "ellipse",
            "--planner",
            "mppi",
            "--proxy",
        ),
        marks=G1,
    ),
    pytest.param(
        ("examples/mujoco/g1_scramble.py", "--robot", "amo", "--planner", "mppi", "--torso"),
        marks=G1,
    ),
    pytest.param(
        ("examples/mujoco/g1_scramble.py", "--robot", "groot", "--planner", "mppi"), marks=G1
    ),
    pytest.param(
        ("examples/mujoco/g1_scramble_social_eval.py", "--configs", "mppi", "--seeds", "0"),
        marks=G1,
    ),
    pytest.param("examples/mujoco/g1_standup.py", marks=G1),
    pytest.param("examples/mujoco/g1_navigate.py", marks=G1),
    pytest.param("examples/mujoco/g1_amo_demo.py", marks=G1),
    pytest.param(("examples/mujoco/g1_corridor.py", "--footprint", "ellipse"), marks=G1),
    pytest.param(
        ("examples/mujoco/g1_corridor.py", "--planner", "mppi", "--offset", "-0.35"), marks=G1
    ),
    pytest.param("examples/mujoco/g1_footprint_measure.py", marks=G1),
    pytest.param("examples/mujoco/g1_walk_compare.py", marks=G1),
    pytest.param("examples/mujoco/g1_plaza.py", marks=G1),
    pytest.param("examples/mujoco/g1_model_distance.py", marks=G1),
    pytest.param("examples/mujoco/g1_scramble.py", marks=G1),
]


@pytest.mark.slow
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
