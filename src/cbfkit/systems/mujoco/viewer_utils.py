"""Helpers for opening the MuJoCo viewer from CBFKit examples.

On macOS ``mujoco.viewer.launch_passive`` must own the Cocoa main thread, which
only MuJoCo's ``mjpython`` launcher provides; a plain interpreter raises

    RuntimeError: `launch_passive` requires that the Python script be run
    under `mjpython` on macOS

The ``mujoco`` wheel installs ``mjpython`` next to ``python`` in the same
environment, so :func:`relaunch_under_mjpython_if_needed` simply re-execs the
running script under it -- call it *before* any expensive setup so the JIT
work is not done twice. Adapted from hydrax's ``utils/mjpython.py`` (MIT).

Set ``CBFKIT_NO_MJPYTHON=1`` to disable the relaunch (headless runs).
"""

import os
import sys
from pathlib import Path

import mujoco.viewer

_RELAUNCHED = "CBFKIT_MJPYTHON_RELAUNCHED"
_OPT_OUT = "CBFKIT_NO_MJPYTHON"


def under_mjpython() -> bool:
    """True if this process can open a passive viewer on macOS (or is not on macOS)."""
    if sys.platform != "darwin":
        return True
    # mjpython installs an ``_MjPythonBase`` here at startup; this is exactly
    # what ``launch_passive`` checks.
    return getattr(mujoco.viewer, "_MJPYTHON", None) is not None


def mjpython_executable() -> Path:
    """The ``mjpython`` that belongs to the running interpreter (may not exist)."""
    return Path(sys.executable).with_name("mjpython")


def relaunch_under_mjpython_if_needed() -> None:
    """Re-exec the current script under ``mjpython`` if macOS requires it.

    Does nothing off macOS, when already under ``mjpython``, or when
    ``CBFKIT_NO_MJPYTHON`` is set. Raises ``RuntimeError`` (with a hint) when a
    relaunch is needed but impossible: no ``mjpython`` next to the interpreter,
    or nothing to re-run (REPL, notebook, ``python -c``).
    """
    if under_mjpython() or os.environ.get(_OPT_OUT):
        return

    mjpython = mjpython_executable()
    hint = f"Run the script under MuJoCo's launcher instead: `{mjpython} <script> ...`."

    if os.environ.get(_RELAUNCHED):
        raise RuntimeError(
            f"Relaunched under {mjpython}, but the viewer is still unavailable. {hint}"
        )
    if not mjpython.is_file():
        raise RuntimeError(
            f"The MuJoCo viewer needs `mjpython` on macOS, but there is none next to "
            f"{sys.executable}. Install the `mujoco` wheel in this environment."
        )

    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None:  # started with `python -m some.module`
        args = ["-m", spec.name, *sys.argv[1:]]
    elif Path(sys.argv[0]).is_file():
        args = list(sys.argv)
    else:
        raise RuntimeError(
            f"The MuJoCo viewer needs `mjpython` on macOS, and there is no script to relaunch "
            f"(sys.argv[0] = {sys.argv[0]!r}). {hint}"
        )

    print(f"Relaunching under {mjpython} (macOS viewer requirement)...")
    sys.stdout.flush()
    os.execve(str(mjpython), [str(mjpython), *args], {**os.environ, _RELAUNCHED: "1"})
