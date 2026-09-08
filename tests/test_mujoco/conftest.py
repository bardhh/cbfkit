"""MuJoCo test suite: not collected when the optional extra is not installed.

``pytest.importorskip`` at conftest import time aborts collection outright on
current pytest, so the directory is ignored instead (``pip install cbfkit[mujoco]``).

Tests marked ``g1_mjx`` compile the Unitree G1 MJX step (policy in the loop).
That compile peaks at about 20 GB of resident memory on CPU XLA (measured), which
kills a 16 GB GitHub runner outright, so they are skipped on hosts below
``G1_MJX_MIN_RAM_GB``.
"""

import os

import pytest

try:
    import mujoco  # noqa: F401
except ImportError:  # optional extra absent
    collect_ignore_glob = ["test_*.py"]

G1_MJX_MIN_RAM_GB = 24.0


def total_ram_gb():
    """Physical memory in GiB, or +inf when the platform does not report it."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (AttributeError, ValueError, OSError):
        return float("inf")


def pytest_collection_modifyitems(config, items):
    ram = total_ram_gb()
    if ram >= G1_MJX_MIN_RAM_GB:
        return
    skip = pytest.mark.skip(
        reason=f"G1 MJX step compile needs ~20 GB RSS; this host reports {ram:.0f} GB"
    )
    for item in items:
        if item.get_closest_marker("g1_mjx") is not None:
            item.add_marker(skip)
