"""Missing platform-specific solvers should fail on use, not module import."""

import importlib
import platform
import runpy
from pathlib import Path

import jax.numpy as jnp
import pytest

from cbfkit.optimization.quadratic_program import qp_solver_cvxopt


@pytest.mark.parametrize(
    "machine,package", [("arm64", "kvxopt"), ("aarch64", "kvxopt"), ("x86_64", "cvxopt")]
)
def test_missing_backend_reports_install_extra(monkeypatch, machine, package):
    missing = ImportError(f"No module named {package}")

    def unavailable(name):
        assert name == package
        raise missing

    with monkeypatch.context() as patch:
        patch.setattr(platform, "machine", lambda: machine)
        patch.setattr(importlib, "import_module", unavailable)
        namespace = runpy.run_path(str(Path(qp_solver_cvxopt.__file__)))

    with pytest.raises(ImportError, match=r"pip install 'cbfkit\[cvxopt\]'") as exc:
        namespace["solve"](jnp.eye(1), jnp.zeros(1))
    assert exc.value.__cause__ is missing
