import os
import subprocess
import sys

import pytest


ALIASES = [
    ("tutorials/mppi_cbf_reach_avoid.py", "tutorials/simulate_mppi_cbf.py"),
    ("tutorials/mppi_stl_reach_avoid.py", "tutorials/simulate_mppi_stl.py"),
    (
        "tutorials/single_integrator_dynamic_obstacles.py",
        "tutorials/single_integrator_reach_avoid_dyn_obs.py",
    ),
]


@pytest.mark.parametrize("alias_path,target_path", ALIASES)
def test_tutorial_alias_prints_mapping(alias_path: str, target_path: str) -> None:
    # Newer bolt revisions may keep these scripts as first-class entrypoints
    # (no deprecation shim). In that case this compatibility check is not applicable.
    if not os.path.exists(target_path):
        pytest.skip(f"Target entrypoint {target_path} is not present on this branch.")

    with open(alias_path, "r", encoding="utf-8") as f:
        alias_source = f.read().lower()
    if "deprecated" not in alias_source:
        pytest.skip(f"{alias_path} is not a deprecation shim on this branch.")

    env = os.environ.copy()
    env["CBFKIT_TUTORIAL_ALIAS_ONLY"] = "1"
    env["PYTHONPATH"] = f"{os.getcwd()}{os.pathsep}{os.path.join(os.getcwd(), 'src')}"

    proc = subprocess.run(
        [sys.executable, alias_path],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "deprecated" in proc.stdout.lower()
    assert target_path in proc.stdout
