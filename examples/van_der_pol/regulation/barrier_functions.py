"""Compatibility re-exports for regulation-specific import paths.

Use `examples.van_der_pol.common.barrier_functions` as the canonical source.
"""

import os
import sys

# Add the project root to the path so the `examples` package resolves when this
# module is executed directly as a script (not just imported from the repo root).
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from examples.van_der_pol.common.barrier_functions import (  # noqa: F401
    cbf,
    obstacle_ff,
    obstacle_ff_barriers,
)

__all__ = ["cbf", "obstacle_ff", "obstacle_ff_barriers"]
