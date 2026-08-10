"""Compatibility re-exports for regulation-specific import paths.

Use `examples.van_der_pol.common.lyapunov_functions` as the canonical source.
"""

import os
import sys

# Add the project root to the path so the `examples` package resolves when this
# module is executed directly as a script (not just imported from the repo root).
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from examples.van_der_pol.common.lyapunov_functions import (  # noqa: F401
    fxts_lyapunov,
    fxts_lyapunov_conditions,
)

__all__ = ["fxts_lyapunov", "fxts_lyapunov_conditions"]
