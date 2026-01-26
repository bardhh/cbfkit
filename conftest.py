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
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (won't override existing env vars)
load_dotenv(Path(__file__).parent / ".env")


def pytest_configure(config):
    """Configure pytest cache directory from environment variable."""
    cache_dir = os.environ.get("PYTEST_CACHE_DIR")
    if cache_dir:
        config.cache._cachedir = Path(cache_dir)
