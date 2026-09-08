"""MuJoCo test suite: not collected when the optional extra is not installed.

``pytest.importorskip`` at conftest import time aborts collection outright on
current pytest, so the directory is ignored instead (``pip install cbfkit[mujoco]``).
"""

try:
    import mujoco  # noqa: F401
except ImportError:  # optional extra absent
    collect_ignore_glob = ["test_*.py"]
