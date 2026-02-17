#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WITH_DEPENDENCY_CONTRACT=0
WITH_SLOW=0

for arg in "$@"; do
  case "$arg" in
    --with-dependency-contract)
      WITH_DEPENDENCY_CONTRACT=1
      ;;
    --with-slow)
      WITH_SLOW=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--with-dependency-contract] [--with-slow]" >&2
      exit 2
      ;;
  esac
done

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/5] Enforcing .jules artifact policy..."
bash scripts/check_jules_policy.sh

echo "[2/5] Compile-checking merge-gate tests..."
python -m py_compile \
  tests/test_controllers/test_mppi_parity_and_perf.py \
  tests/test_quadrotor_geometric_controller.py \
  tests/test_ros_controller_wrappers_smoke.py \
  tests/test_rectify_relative_degree_compatibility.py \
  tests/test_simulation/test_error_reporting.py \
  tests/test_simulation/test_solver_status_policy.py \
  tests/test_utils/test_visualization_optional_dependency.py \
  tests/test_tutorial_migration_aliases.py

echo "[3/5] Running targeted merge-gate tests..."
pytest -q \
  tests/test_controllers/test_mppi_parity_and_perf.py \
  tests/test_quadrotor_geometric_controller.py \
  tests/test_ros_controller_wrappers_smoke.py \
  tests/test_rectify_relative_degree_compatibility.py \
  tests/test_simulation/test_error_reporting.py \
  tests/test_simulation/test_solver_status_policy.py \
  tests/test_utils/test_visualization_optional_dependency.py \
  tests/test_tutorial_migration_aliases.py

if [[ "$WITH_DEPENDENCY_CONTRACT" -eq 1 ]]; then
  echo "[4/5] Validating optional dependency contracts..."
  python scripts/validate_optional_dependency_contracts.py
else
  echo "[4/5] Skipped optional dependency contract checks."
fi

if [[ "$WITH_SLOW" -eq 1 ]]; then
  echo "[5/5] Running slow example/tutorial checks..."
  pytest -q -m "slow" tests/test_examples_and_tutorials.py tests/test_differential_drive_examples.py
else
  echo "[5/5] Skipped slow example/tutorial checks."
fi

echo "Pre-merge gate checks passed."
