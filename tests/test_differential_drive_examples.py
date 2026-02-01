import os
import subprocess
import sys

import pytest

SCRIPTS_TO_TEST = [
    "examples/differential_drive/single_robot_cbf.py",
    "examples/differential_drive/barrier_activated_cbf.py",
]


@pytest.mark.slow
@pytest.mark.parametrize("script_path", SCRIPTS_TO_TEST)
def test_differential_drive_example_execution(script_path):
    """Runs the specified example script as a subprocess and asserts it exits with code 0."""
    if not os.path.exists(script_path):
        pytest.fail(f"Script not found: {script_path}")

    # Set environment variables to force headless mode for matplotlib
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = f"{os.getcwd()}{os.pathsep}{os.path.join(os.getcwd(), 'src')}"

    try:
        subprocess.run(
            [sys.executable, script_path], env=env, capture_output=True, text=True, check=True
        )

    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Script {script_path} failed with exit code {e.returncode}.\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}"
        )
