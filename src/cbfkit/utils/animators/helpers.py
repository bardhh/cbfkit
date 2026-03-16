"""Shared helper utilities for animation backends."""

from pathlib import Path
from typing import Optional

import numpy as np

from .config import AnimationConfig, DEFAULT_CONFIG
from .deps import _require_matplotlib

# Matplotlib short color names / tab palette -> CSS hex for Plotly
_MPL_TO_CSS: dict = {
    "r": "red",
    "g": "green",
    "b": "blue",
    "k": "black",
    "w": "white",
    "c": "cyan",
    "m": "magenta",
    "y": "yellow",
    "tab:blue": "#1f77b4",
    "tab:orange": "#ff7f0e",
    "tab:green": "#2ca02c",
    "tab:red": "#d62728",
    "tab:purple": "#9467bd",
    "tab:brown": "#8c564b",
    "tab:pink": "#e377c2",
    "tab:gray": "#7f7f7f",
    "tab:olive": "#bcbd22",
    "tab:cyan": "#17becf",
}


def _to_css_color(color: str) -> str:
    """Convert a matplotlib color string to a CSS-compatible color."""
    return _MPL_TO_CSS.get(color, color)


def _ellipse_svg_path(cx: float, cy: float, rx: float, ry: float) -> str:
    """Return an SVG path string for an axis-aligned ellipse.

    Uses four cubic Bezier segments (the standard circle-to-Bezier
    approximation scaled to ``rx``, ``ry``).  Plotly layout shapes do
    **not** support SVG arc (``A``) commands, so Bezier curves are required.
    """
    k = 0.5522847498
    kx, ky = rx * k, ry * k
    return (
        f"M {cx},{cy - ry} "
        f"C {cx + kx},{cy - ry} {cx + rx},{cy - ky} {cx + rx},{cy} "
        f"C {cx + rx},{cy + ky} {cx + kx},{cy + ry} {cx},{cy + ry} "
        f"C {cx - kx},{cy + ry} {cx - rx},{cy + ky} {cx - rx},{cy} "
        f"C {cx - rx},{cy - ky} {cx - kx},{cy - ry} {cx},{cy - ry} Z"
    )


def _get_fading_segments(x, y):
    """Create line segments and per-segment alpha values for a fading line.

    Parameters
    ----------
    x, y : array-like
        Coordinate sequences.

    Returns
    -------
    segments : list of ndarray
        Each element is a ``(2, 2)`` array ``[[x0, y0], [x1, y1]]``.
    alphas : ndarray
        Alpha values from 1.0 (start) to 0.0 (end).
    """
    if len(x) < 2:
        return [], []
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    alphas = np.linspace(1.0, 0.0, len(segments))
    return segments, alphas


def _circle_shape(cx: float, cy: float, r: float, color: str,
                  alpha: float = 1.0, dash: str = "solid",
                  fill: bool = True) -> dict:
    """Build a Plotly layout circle shape dict."""
    css = _to_css_color(color)
    if fill:
        fillcolor = css
    else:
        fillcolor = "rgba(0,0,0,0)"
    return dict(
        type="circle",
        xref="x", yref="y",
        x0=cx - r, y0=cy - r,
        x1=cx + r, y1=cy + r,
        line=dict(color=css, dash=dash, width=1),
        fillcolor=fillcolor,
        opacity=alpha,
    )


# Number of opacity buckets for Plotly fading-line approximation
_PLOTLY_FADE_BUCKETS = 4

# Scatter overlay opacity factor (relative to trajectory alpha)
_SCATTER_ALPHA_FACTOR = 0.55


def _compute_plotly_frame_step(dt: float, n_total: int, max_frames: int = 200,
                               min_frame_ms: float = 40):
    """Compute downsampled frame indices and duration for Plotly animations.

    Returns ``(frame_indices, frame_duration_ms)``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    min_step = max(1, int(np.ceil(min_frame_ms / (dt * 1000))))
    step = max(min_step, n_total // max_frames)
    frame_indices = list(range(0, n_total, step))
    if frame_indices[-1] != n_total - 1:
        frame_indices.append(n_total - 1)
    frame_duration_ms = dt * step * 1000
    return frame_indices, frame_duration_ms


def _plotly_animation_controls(frames, frame_duration_ms: float,
                               button_y: float = -0.32,
                               slider_y: float = -0.08):
    """Build Plotly ``updatemenus`` (play/pause) and ``sliders`` dicts."""
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            y=button_y, x=0.5,
            xanchor="center", yanchor="top",
            direction="left",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="\u25b6 Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_duration_ms, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        ),
                    ],
                ),
                dict(
                    label="\u23f8 Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )
    ]

    sliders = [
        dict(
            active=0,
            steps=[
                dict(
                    args=[
                        [f.name],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                    label=f.name,
                    method="animate",
                )
                for f in frames
            ],
            x=0.1, len=0.8,
            y=slider_y, yanchor="top",
            currentvalue=dict(prefix="Time: ", visible=True, xanchor="center"),
            transition=dict(duration=0),
        )
    ]

    return updatemenus, sliders


def save_animation(
    anim,
    path: str,
    config: Optional[AnimationConfig] = None,
) -> str:
    """Save a matplotlib ``FuncAnimation`` with automatic format fallback.

    Tries MP4 (via *ffmpeg*) first, then falls back to GIF (via *pillow*).

    Parameters
    ----------
    anim : FuncAnimation
        The animation object to save.
    path : str
        Desired output path.  The suffix is replaced as needed.
    config : AnimationConfig, optional
        Overrides for fps / dpi / bitrate.  Uses :data:`DEFAULT_CONFIG` when
        *None*.

    Returns
    -------
    str
        Absolute path of the saved file, or ``""`` on failure.
    """
    _require_matplotlib()
    cfg = config or DEFAULT_CONFIG
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try MP4 via ffmpeg
    mp4_path = output_path.with_suffix(".mp4")
    try:
        anim.save(
            str(mp4_path),
            writer="ffmpeg",
            fps=cfg.fps,
            dpi=cfg.dpi,
            bitrate=cfg.bitrate,
        )
        abs_path = str(mp4_path.resolve())
        print(f"\nAnimation saved to: file://{abs_path}")
        return abs_path
    except Exception:
        pass

    # Fallback to GIF via pillow
    gif_path = output_path.with_suffix(".gif")
    try:
        anim.save(str(gif_path), writer="pillow", fps=cfg.fps, dpi=cfg.dpi)
        abs_path = str(gif_path.resolve())
        print(f"\nAnimation saved to: file://{abs_path}")
        return abs_path
    except Exception as exc:
        print(f"Could not save animation: {exc}")
        return ""
