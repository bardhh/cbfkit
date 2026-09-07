"""Helpers for replaying, rendering and viewing MuJoCo simulations from CBFKit examples.

Replay/render: :func:`replay_states` (forward kinematics over logged flat states),
:func:`add_marker` (visual-only geoms in an ``mjvScene``), :func:`render_gif` (offscreen
tracking-camera GIF) and :func:`replay_in_viewer` (passive viewer loop). All accept an
optional ``markers(scn, k, t)`` callback so time-varying overlays (moving obstacles,
keep-out rings, goals) can be drawn per frame.

Viewer launch on macOS:

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
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

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


# --------------------------------------------------------------------------- replay / render
MarkerFn = Callable[[mujoco.MjvScene, int, float], None]


def replay_states(plant, states) -> Iterator[Tuple[mujoco.MjData, int]]:
    """Yield ``(MjData, k)`` with kinematics forwarded at each logged flat state ``states[k]``.

    The same ``MjData`` object is reused between iterations; copy what you need.
    """
    m = plant.mj_model
    d = mujoco.MjData(m)
    states = np.asarray(states)
    for k in range(states.shape[0]):
        d.qpos[:] = states[k, : plant.nq]
        d.qvel[:] = states[k, plant.nq : plant.nq + plant.nv]
        mujoco.mj_forward(m, d)
        yield d, k


def add_marker(scn: mujoco.MjvScene, geom_type, size, pos, rgba) -> bool:
    """Append a visual-only geom (identity rotation) to ``scn``; False if the scene is full."""
    if scn.ngeom >= scn.maxgeom:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        geom_type,
        np.asarray(size, float),
        np.asarray(pos, float),
        np.eye(3).flatten(),
        np.asarray(rgba, np.float32),
    )
    scn.ngeom += 1
    return True


def render_gif(
    plant,
    states,
    path: str,
    *,
    track_body: int,
    markers: Optional[MarkerFn] = None,
    fps: int = 25,
    width: int = 640,
    height: int = 360,
    distance: float = 4.5,
    azimuth: float = 135.0,
    elevation: float = -25.0,
) -> Optional[str]:
    """Render logged states to a GIF with a camera tracking body ``track_body``.

    ``markers(scn, k, t)`` may add visual geoms per rendered frame (``t = k * plant.dt``).
    Returns ``path``, or None (after printing why) when offscreen rendering is unavailable.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import animation
    from matplotlib import pyplot as plt
    from matplotlib.artist import Artist

    try:
        renderer = mujoco.Renderer(plant.mj_model, height=height, width=width)
    except Exception as exc:  # noqa: BLE001
        print(f"offscreen rendering unavailable ({exc}); skipping GIF")
        return None
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = int(track_body)
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    every = max(1, int(round(1.0 / (fps * plant.dt))))
    frames = []
    for d, k in replay_states(plant, states):
        if k % every:
            continue
        renderer.update_scene(d, camera=cam)
        if markers is not None:
            markers(renderer.scene, k, k * plant.dt)
        frames.append(renderer.render().copy())
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def _draw(i: int) -> Tuple[Artist, ...]:
        im.set_data(frames[i])
        return (im,)

    anim = animation.FuncAnimation(fig, _draw, frames=len(frames), interval=1000 / fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {path}")
    return path


def replay_in_viewer(plant, states, markers: Optional[MarkerFn] = None) -> None:
    """Loop the logged states in the passive viewer (needs ``mjpython`` on macOS)."""
    if not under_mjpython():
        print("viewer skipped: needs mjpython (rerun with --view; it relaunches)")
        return
    m = plant.mj_model
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            for dd, k in replay_states(plant, states):
                if not viewer.is_running():
                    break
                d.qpos[:] = dd.qpos
                d.qvel[:] = dd.qvel
                mujoco.mj_forward(m, d)
                viewer.user_scn.ngeom = 0
                if markers is not None:
                    markers(viewer.user_scn, k, k * plant.dt)
                viewer.sync()
                time.sleep(plant.dt)
