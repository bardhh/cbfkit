"""
Root conftest.py - test-session environment setup.

pytest.ini anchors the rootdir here, so pytest imports this before any test
module.  That makes it the earliest available hook for process-wide setup:

1. Loads a .env file from the project root via python-dotenv, when both the
   file and the package are present.  Existing environment variables win.
2. Puts ./src at the front of sys.path, so the checkout is exercised rather
   than an unrelated cbfkit sitting in site-packages.
3. Defaults JAX to the CPU backend, which stops sandboxed and CI hosts from
   crashing on a Metal/GPU backend that is visible but not usable.
4. Honours PYTEST_CACHE_DIR, for keeping the cache off a synced folder.

Note the reach of step 1: .env only affects code running inside this process.
mypy, ruff and black are separate processes and never see it.
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
