import ast
import os
import shutil
import subprocess
import sys

import pytest

# List of script paths relative to the project root
# These are the scripts we want to verify run successfully (exit code 0)
SCRIPTS_TO_TEST = [
    "examples/unicycle/reach_goal/mppi_cbf.py",
    "examples/unicycle/reach_goal/stochastic_cbf.py",
    "examples/unicycle/reach_goal/vanilla_cbf.py",
    # "examples/unicycle/reach_goal/risk_aware_cbf.py",  # Disabled: needs visualization fix
    "tutorials/unicycle_reach_avoid.py",
    "tutorials/single_integrator_dynamic_obstacles.py",
    "tutorials/mppi_stl_reach_avoid.py",
    "tutorials/mppi_cbf_reach_avoid.py",
    "tutorials/mppi_stochastic_cbf_reach_avoid.py",
    "tutorials/code_generation_example.py",
]


def _assert_script_has_executable_code(script_path: str) -> None:
    """Guard against silently passing placeholder scripts."""
    with open(script_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=script_path)
    nodes = list(tree.body)

    # Ignore module-level docstring when checking whether the script is real code.
    if (
        nodes
        and isinstance(nodes[0], ast.Expr)
        and isinstance(nodes[0].value, ast.Constant)
        and isinstance(nodes[0].value.value, str)
    ):
        nodes = nodes[1:]

    executable_nodes = [node for node in nodes if not isinstance(node, ast.Pass)]
    if not executable_nodes:
        pytest.fail(
            f"Script {script_path} has no executable code (docstring/comment placeholder)."
        )


@pytest.mark.slow
@pytest.mark.parametrize("script_path", SCRIPTS_TO_TEST)
def test_example_script_execution(script_path, tmp_path):
    """Runs the specified example script as a subprocess and asserts it exits with code 0."""
    # Check if file exists
    if not os.path.exists(script_path):
        pytest.fail(f"Script not found: {script_path}")
    _assert_script_has_executable_code(script_path)

    # Set environment variables to force headless mode for matplotlib
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / ".mplconfig")
    env["CBFKIT_TEST_MODE"] = "1"
    # Force CPU backend in subprocesses to avoid Metal/GPU initialization aborts
    # in CI/sandboxed environments where no visible accelerator is present.
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["JAX_PLATFORMS"] = "cpu"
    # Ensure tmp_path (for generated/copied modules), src and root (for examples) are in python path
    # Prepend tmp_path to PYTHONPATH so that imported modules (like 'tutorials') are loaded
    # from the temporary directory (where code generation happens) instead of the source tree.
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{os.getcwd()}{os.pathsep}{os.path.join(os.getcwd(), 'src')}"

    # Copy script parent directory to tmp_path to ensure relative assets/imports work
    # and to isolate output files.
    src_dir = os.path.dirname(script_path)
    dst_dir = tmp_path / src_dir

    # We use dirs_exist_ok=True just in case, though tmp_path should be empty
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    try:
        # Run the script
        # We use sys.executable to ensure we use the same python interpreter (venv)
        subprocess.run(
            [sys.executable, script_path],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )

    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Script {script_path} failed with exit code {e.returncode}.\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}"
        )
