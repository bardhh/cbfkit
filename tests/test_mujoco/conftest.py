"""MuJoCo test suite: skipped wholesale when the optional extra is not installed."""

import pytest

mujoco = pytest.importorskip("mujoco", reason="pip install cbfkit[mujoco]")
