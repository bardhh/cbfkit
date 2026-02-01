import os
import subprocess
import sys

import pytest

# List of script paths relative to the project root
# These are the scripts we want to verify run successfully (exit code 0)
SCRIPTS_TO_TEST = [
    "examples/unicycle/reach_goal/mppi_cbf_control.py",
    "examples/unicycle/reach_goal/stochastic_cbf_control.py",
    "examples/unicycle/reach_goal/vanilla_cbf_control.py",
    # "examples/unicycle/reach_goal/risk_aware_cbf_control.py",  # Disabled: needs visualization fix
    "tutorials/unicycle_reach_avoid.py",
    "tutorials/single_integrator_dynamic_obstacles.py",
    # "tutorials/mppi_stl_reach_avoid.py", # Requires complex setup or long run?
    "tutorials/mppi_cbf_reach_avoid.py",
    "tutorials/mppi_stochastic_cbf_reach_avoid.py",
    "tutorials/code_generation_example.py",
    "tutorials/multi_robot_double_integrator.py",
]


@pytest.mark.parametrize("script_path", SCRIPTS_TO_TEST)
def test_example_script_execution(script_path):
    """Runs the specified example script as a subprocess and asserts it exits with code 0."""
    # Check if file exists
    if not os.path.exists(script_path):
        pytest.fail(f"Script not found: {script_path}")

    # Set environment variables to force headless mode for matplotlib
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src")  # Ensure src is in python path

    try:
        # Run the script
        # We use sys.executable to ensure we use the same python interpreter (venv)
        subprocess.run(
            [sys.executable, script_path], env=env, capture_output=True, text=True, check=True
        )

    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Script {script_path} failed with exit code {e.returncode}.\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}"
        )
