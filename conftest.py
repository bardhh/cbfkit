"""
Root conftest.py - Loads environment variables from .env file.

This module:
1. Uses python-dotenv to load .env file (if present)
2. Sets environment variables for mypy, ruff, pytest, and Python bytecode caches
3. Sets JAX/XLA GPU memory configuration if specified
4. Configures pytest's cache directory dynamically

Environment variables are only set if not already present in the environment,
allowing system-level overrides.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv is optional: tests should still run when environment loading support
    # is not installed in the current interpreter.
    def load_dotenv(*_args, **_kwargs):
        return False

# Load .env file from project root (won't override existing env vars)
load_dotenv(Path(__file__).parent / ".env")

# Ensure local src/ package is tested, not an unrelated site-packages install.
_ROOT_DIR = Path(__file__).parent
_SRC_DIR = _ROOT_DIR / "src"
if _SRC_DIR.exists():
    src_str = str(_SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

# Default test execution to CPU JAX unless explicitly overridden.
# This prevents hard crashes on hosts where Metal/GPU backends are visible
# but not usable in sandboxed/CI environments.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def pytest_configure(config):
    """Configure pytest cache directory from environment variable."""
    cache_dir = os.environ.get("PYTEST_CACHE_DIR")
    if cache_dir:
        config.cache._cachedir = Path(cache_dir)
