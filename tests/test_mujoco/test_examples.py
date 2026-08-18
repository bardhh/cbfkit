"""Smoke-run the MuJoCo examples in test mode (short horizon, no plots)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ["examples/mujoco/cart_pole_swingup.py"]


@pytest.mark.parametrize("script", EXAMPLES)
def test_example_runs_in_test_mode(script):
    env = dict(os.environ, CBFKIT_TEST_MODE="1", MPLBACKEND="Agg", JAX_PLATFORM_NAME="cpu")
    proc = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "mean upright-distance" in proc.stdout
