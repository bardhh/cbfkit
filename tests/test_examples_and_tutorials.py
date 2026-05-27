import ast
import os
import shutil
import subprocess
import sys

import pytest

# List of script paths relative to the project root
# These are the scripts we want to verify run successfully (exit code 0)
SCRIPTS_TO_TEST = [
    # Unicycle examples
    "examples/unicycle/reach_goal/mppi_cbf.py",
    "examples/unicycle/reach_goal/mppi_stochastic_cbf.py",
    "examples/unicycle/reach_goal/stochastic_cbf.py",
    "examples/unicycle/reach_goal/vanilla_cbf.py",
    "examples/unicycle/reach_goal/unicycle_reach_avoid_cbf.py",
    "examples/unicycle/reach_goal/vanilla_cbf_accel_unicycle.py",
    "examples/unicycle/reach_goal/risk_aware_cbf_monte_carlo.py",
    # "examples/unicycle/reach_goal/risk_aware_cbf.py",  # Disabled: needs visualization fix
    # Differential drive examples
    "examples/differential_drive/obstacle_avoidance/single_robot_cbf.py",
    "examples/differential_drive/obstacle_avoidance/dynamic_obstacle_cbf.py",
    "examples/differential_drive/obstacle_avoidance/augmented_dynamic_obstacle_cbf.py",
    "examples/differential_drive/obstacle_avoidance/barrier_activated_cbf.py",
    "examples/differential_drive/human_aware_navigation/mppi_cbf.py",
    "examples/differential_drive/human_aware_navigation/multi_scenario_comparison.py",
    # Single integrator examples
    "examples/single_integrator/reach_goal/perfect_sensing.py",
    "examples/single_integrator/reach_goal/ekf.py",
    "examples/single_integrator/reach_goal/ukf.py",
    # Pedestrian examples
    "examples/pedestrian/navigate_among_pedestrians/head_on.py",
    # Fixed-wing examples
    "examples/fixed_wing/reach_drop_point/ekf.py",
    # Neural CBF examples
    "examples/neural_cbf/neural_cbf_obstacle_avoidance.py",
    # Tutorials
    "tutorials/single_integrator_dynamic_obstacles.py",
    "tutorials/mppi_stl_reach_avoid.py",
    "tutorials/mppi_cbf_reach_avoid.py",
    "tutorials/code_generation_example.py",
    "tutorials/multi_robot_coordination_codegen.py",
    "tutorials/multi_robot_3d_reachavoid.py",
    # Risk-aware comparison
    "examples/single_integrator/risk_aware_comparison/run_comparison.py",
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
        pytest.fail(f"Script {script_path} has no executable code (docstring/comment placeholder).")


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["jaxopt", "fast"])
@pytest.mark.parametrize("script_path", SCRIPTS_TO_TEST)
def test_example_script_execution(script_path, solver, tmp_path):
    """Runs the specified example script as a subprocess and asserts it exits with code 0.

    Parametrized over solvers: each script must pass under both the default
    jaxopt OSQP and the fast PDIPM backend. The subprocess reads
    CBFKIT_QP_SOLVER from its environment; get_solver() honors it.
    """
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
    env["CBFKIT_QP_SOLVER"] = solver
    # Ensure tmp_path (for generated/copied modules), src and root (for examples) are in python path
    # Prepend tmp_path to PYTHONPATH so that imported modules (like 'tutorials') are loaded
    # from the temporary directory (where code generation happens) instead of the source tree.
    env[
        "PYTHONPATH"
    ] = f"{tmp_path}{os.pathsep}{os.getcwd()}{os.pathsep}{os.path.join(os.getcwd(), 'src')}"

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


def test_risk_aware_comparison_ordering():
    """ACC 2026 Fig. 1: the four controllers separate in the expected p_fail order.

    Asserts the Fig. 1 ordering -- nominal > RA-CBF-CT > RA-CBF-DT >= S-CBF -- using a small,
    fixed-seed rollout count (exact magnitudes require the full 1000-rollout run). The CT
    variant rides near the boundary and fails at ~rho_d; the more-conservative DT margin and
    the supermartingale S-CBF stay safer."""
    from examples.single_integrator.risk_aware_comparison.run_comparison import run_all

    res = run_all(n_trials=64, seed=0)  # fixed seed => deterministic CI
    p = {k: v[0] for k, v in res.items()}
    assert p["nominal"] > 0.9, p  # unfiltered outward drive almost always exits
    assert 0.03 < p["ra_cbf_ct"] < 0.30, p  # RA-CBF-CT rides near boundary, ~rho_d
    assert p["ra_cbf_dt"] < p["ra_cbf_ct"], p  # DT margin is more conservative than CT
    assert p["s_cbf"] <= p["ra_cbf_dt"] + 1e-9, p  # S-CBF is the safest controller (Fig. 1)
    for cbf in ("s_cbf", "ra_cbf_dt", "ra_cbf_ct"):
        assert p[cbf] < p["nominal"], p
