#!/usr/bin/env python3
"""Validate optional dependency contracts for user-facing error guidance.

This script intentionally avoids importing modules through package ``__init__``
paths that may require unrelated optional dependencies (for example, ``jaxopt``).
The contract checks here only target user-facing guidance for explicitly optional
features such as CasADi and visualization.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import tempfile
from pathlib import Path

import jax.numpy as jnp

# Ensure matplotlib/fontconfig can write cache files in restricted environments.
_cache_root = Path(tempfile.gettempdir()) / "cbfkit_optional_dep_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
mpl_cache = _cache_root / "mpl"
xdg_cache = _cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import cbfkit.utils.visualization as visualization


def _load_qp_solver_casadi_module():
    """Load qp_solver_casadi without importing parent package __init__ files."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "cbfkit"
        / "optimization"
        / "quadratic_program"
        / "qp_solver_casadi.py"
    )
    namespace = runpy.run_path(str(module_path))
    return namespace


def _check_casadi_contract() -> None:
    if importlib.util.find_spec("casadi") is not None:
        return

    qp_solver_casadi_ns = _load_qp_solver_casadi_module()
    solve = qp_solver_casadi_ns["solve"]

    try:
        solve(jnp.eye(1), jnp.zeros((1,)))
    except ImportError as exc:
        if "cbfkit[casadi]" not in str(exc):
            raise AssertionError(
                "CasADi missing-path error must include installation guidance "
                "for `cbfkit[casadi]`."
            ) from exc
    else:
        raise AssertionError("Expected ImportError with installation guidance for CasADi.")


def _check_visualization_contract() -> None:
    if visualization.HAS_MATPLOTLIB:
        return

    try:
        visualization.require_visualization()
    except ImportError as exc:
        if "cbfkit[vis]" not in str(exc):
            raise AssertionError(
                "Visualization missing-path error must include installation guidance "
                "for `cbfkit[vis]`."
            ) from exc
    else:
        raise AssertionError(
            "Expected ImportError with installation guidance for visualization."
        )


def main() -> None:
    _check_casadi_contract()
    _check_visualization_contract()
    print("optional dependency contract checks passed")


if __name__ == "__main__":
    main()
