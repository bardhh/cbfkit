"""Reusable animation framework for CBFKit simulations.

Provides :class:`CBFAnimator` for building 2D trajectory animations with a
declarative API, :class:`AnimationConfig` for centralised defaults, and
:func:`save_animation` for robust file output with format fallback.

Supports two backends:

* ``"matplotlib"`` — generates MP4 / GIF animations via ``FuncAnimation``.
* ``"plotly"`` (default) — generates interactive HTML files with play/pause
  controls and a timeline slider.  Requires ``pip install cbfkit[plotly]``.

Example
-------
>>> from cbfkit.utils.animator import CBFAnimator
>>> anim = CBFAnimator(states, dt=0.1, x_lim=(-4, 4), y_lim=(-4, 4))
>>> anim.add_goal(desired_state[:2], radius=0.25)
>>> anim.add_obstacles(obstacle_centers, ellipsoid_radii=ellipsoid_radii)
>>> anim.add_trajectory(x_idx=0, y_idx=1, label="Trajectory")
>>> anim.save("results/animation.html")

Crowd / multi-agent example:

>>> anim = CBFAnimator(states, dt=0.1, aspect="equal")
>>> anim.add_goal(robot_goal[:2], radius=0.4, color="green")
>>> anim.add_agent(x_idx=0, y_idx=1, body_radius=0.3, body_color="blue",
...                label="Robot")
>>> anim.add_agent(x_idx=4, y_idx=5, body_radius=0.25, body_color="red",
...                zone_radius=1.0, trail_style="--", label="Ped 0")
>>> anim.add_prediction(source="linear", agent_x_idx=4, agent_y_idx=5,
...                     agent_vx_idx=6, agent_vy_idx=7, color="red")
>>> anim.show_time()
>>> anim.auto_limits()
>>> anim.save("results/crowd.html")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

try:
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Ellipse

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def _require_matplotlib():
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "Optional dependency 'matplotlib' not found. "
            "Please install cbfkit[vis] to use visualization features."
        )


def _require_plotly():
    if not _HAS_PLOTLY:
        raise ImportError(
            "Optional dependency 'plotly' not found. "
            "Please install cbfkit[plotly] to use the Plotly backend."
        )


# Matplotlib short color names / tab: palette -> CSS hex for Plotly
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


# ---------------------------------------------------------------------------
# Animation configuration
# ---------------------------------------------------------------------------


@dataclass
class AnimationConfig:
    """Centralised animation parameter defaults.

    Attributes
    ----------
    fps : int
        Frames per second for saved files.
    dpi : int
        Resolution (dots per inch) for saved files.
    interval : int
        Milliseconds between frames for interactive display.
    bitrate : int
        Bitrate for MP4 output (kbps).
    figsize : tuple of float
        Default figure size ``(width, height)`` in inches.
    grid_alpha : float
        Transparency of the background grid.
    blit : bool
        Whether to use blitting for faster rendering.
    plotly_max_frames : int
        Maximum number of frames for the Plotly backend.  Long simulations
        are downsampled to this count to keep the HTML file responsive.
    """

    fps: int = 20
    dpi: int = 100
    interval: int = 50
    bitrate: int = 1800
    figsize: Tuple[float, float] = (10, 8)
    grid_alpha: float = 0.3
    blit: bool = True
    plotly_max_frames: int = 200


DEFAULT_CONFIG = AnimationConfig()
"""Module-level default :class:`AnimationConfig` instance."""


# ---------------------------------------------------------------------------
# Standalone save helper (matplotlib only)
# ---------------------------------------------------------------------------


def save_animation(
    anim: "mpl_animation.FuncAnimation",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CBFAnimator
# ---------------------------------------------------------------------------


class CBFAnimator:
    """Declarative 2D trajectory animator for CBFKit simulations.

    Build an animation by chaining ``add_*`` calls, then :meth:`save` or
    :meth:`show`.  The class encapsulates figure setup, artist creation,
    the ``init`` / ``update`` callbacks, and the save-with-fallback pattern
    so that individual examples need only describe *what* to draw.

    Parameters
    ----------
    states : np.ndarray
        Simulation state history with shape ``(N, state_dim)``.
    dt : float
        Simulation time step (seconds).
    x_lim, y_lim : tuple of float
        Axis limits.
    title : str
        Plot title.
    aspect : str or None
        Axis aspect ratio (e.g. ``"equal"``).  *None* keeps matplotlib default.
    backend : str
        ``"plotly"`` (default) or ``"matplotlib"``.
    config : AnimationConfig, optional
        Animation parameters.  Uses :data:`DEFAULT_CONFIG` when *None*.
    """

    def __init__(
        self,
        states: np.ndarray,
        dt: float = 0.1,
        x_lim: Tuple[float, float] = (-4, 4),
        y_lim: Tuple[float, float] = (-4, 4),
        title: str = "System Behavior",
        aspect: Optional[str] = None,
        backend: str = "plotly",
        config: Optional[AnimationConfig] = None,
    ):
        if backend not in ("matplotlib", "plotly"):
            raise ValueError(f"Unknown backend {backend!r}. Use 'matplotlib' or 'plotly'.")

        self._backend = backend
        if backend == "matplotlib":
            _require_matplotlib()
        else:
            _require_plotly()

        self._states = np.asarray(states)
        self._dt = dt
        self._x_lim = x_lim
        self._y_lim = y_lim
        self._title = title
        self._aspect = aspect
        self._config = config or AnimationConfig()

        # Element descriptors (populated by add_* methods)
        self._goals: List[dict] = []
        self._obstacles: List[dict] = []
        self._trajectories: List[dict] = []
        self._agents: List[dict] = []
        self._predictions: List[dict] = []
        self._show_time: bool = False
        self._frame_callbacks: List[Callable] = []

        # Built objects (created by build / animate)
        self._fig = None          # matplotlib Figure or Plotly Figure
        self._ax = None           # matplotlib Axes (None for Plotly)
        self._anim = None         # matplotlib FuncAnimation (None for Plotly)
        self._traj_artists: List = []
        self._agent_artists: List = []
        self._prediction_artists: List = []
        self._time_text = None

    # -- declarative element API (all return self for chaining) -------------

    def add_goal(
        self,
        position: Union[np.ndarray, Tuple[float, float], List[float]],
        radius: float = 0.25,
        color: str = "r",
        label: str = "Goal",
    ) -> "CBFAnimator":
        """Add a goal marker (filled dot + dashed circle)."""
        self._goals.append(
            {"position": position, "radius": radius, "color": color, "label": label}
        )
        return self

    def add_obstacle(
        self,
        center: Union[np.ndarray, Tuple[float, float], List[float]],
        radius: Optional[float] = None,
        ellipse_radii: Optional[Tuple[float, float]] = None,
        color: str = "k",
        alpha: float = 0.3,
    ) -> "CBFAnimator":
        """Add a single circular or elliptical obstacle."""
        self._obstacles.append(
            {
                "center": center,
                "radius": radius,
                "ellipse_radii": ellipse_radii,
                "color": color,
                "alpha": alpha,
            }
        )
        return self

    def add_obstacles(
        self,
        centers: List,
        ellipsoid_radii: Optional[List] = None,
        radii: Optional[List[float]] = None,
        color: str = "k",
        alpha: float = 0.3,
    ) -> "CBFAnimator":
        """Add multiple obstacles at once.

        Provide *either* ``ellipsoid_radii`` (list of ``(rx, ry)`` per
        obstacle) for ellipses, or ``radii`` (list of floats) for circles.
        """
        for i, center in enumerate(centers):
            ell = ellipsoid_radii[i] if ellipsoid_radii is not None else None
            r = radii[i] if radii is not None else None
            self.add_obstacle(center, radius=r, ellipse_radii=ell, color=color, alpha=alpha)
        return self

    def add_trajectory(
        self,
        x_idx: int = 0,
        y_idx: int = 1,
        data: Optional[np.ndarray] = None,
        color: str = "tab:blue",
        label: str = "Trajectory",
        style: str = "line",
        linewidth: float = 2.0,
        alpha: float = 1.0,
        zorder: int = 3,
    ) -> "CBFAnimator":
        """Add an animated trajectory line or scatter overlay.

        Parameters
        ----------
        x_idx, y_idx : int
            Column indices into *data* (or *states*) for x / y coordinates.
        data : np.ndarray, optional
            Override data source.  Defaults to the *states* array passed to
            the constructor.
        style : ``"line"`` | ``"scatter"``
            Visual style.
        """
        self._trajectories.append(
            {
                "x_idx": x_idx,
                "y_idx": y_idx,
                "data": data,
                "color": color,
                "label": label,
                "style": style,
                "linewidth": linewidth,
                "alpha": alpha,
                "zorder": zorder,
            }
        )
        return self

    def add_agent(
        self,
        x_idx: int,
        y_idx: int,
        data: Optional[np.ndarray] = None,
        body_radius: float = 0.3,
        body_color: str = "blue",
        body_alpha: float = 0.8,
        zone_radius: Optional[float] = None,
        zone_color: Optional[str] = None,
        zone_alpha: float = 0.15,
        trail: bool = True,
        trail_color: Optional[str] = None,
        trail_alpha: float = 0.5,
        trail_style: str = "-",
        label: str = "Agent",
        zorder: int = 5,
    ) -> "CBFAnimator":
        """Add a moving agent (robot or pedestrian) with body, optional
        safety zone, and optional trail.

        Parameters
        ----------
        x_idx, y_idx : int
            Column indices into *data* (or *states*) for x / y position.
        data : np.ndarray, optional
            Override data source.  Defaults to *states*.
        body_radius : float
            Radius of the body circle in data units.
        zone_radius : float, optional
            If set, draws a transparent safety zone circle around the agent.
        trail : bool
            Whether to draw a growing trail line behind the agent.
        trail_style : str
            ``"-"`` for solid, ``"--"`` for dashed.
        """
        self._agents.append(
            {
                "x_idx": x_idx,
                "y_idx": y_idx,
                "data": data,
                "body_radius": body_radius,
                "body_color": body_color,
                "body_alpha": body_alpha,
                "zone_radius": zone_radius,
                "zone_color": zone_color or body_color,
                "zone_alpha": zone_alpha,
                "trail": trail,
                "trail_color": trail_color or body_color,
                "trail_alpha": trail_alpha,
                "trail_style": trail_style,
                "label": label,
                "zorder": zorder,
            }
        )
        return self

    def add_prediction(
        self,
        source: str = "linear",
        agent_x_idx: Optional[int] = None,
        agent_y_idx: Optional[int] = None,
        agent_vx_idx: Optional[int] = None,
        agent_vy_idx: Optional[int] = None,
        agent_data: Optional[np.ndarray] = None,
        horizon: int = 20,
        trajectory_data: Optional[List[np.ndarray]] = None,
        traj_x_row: int = 0,
        traj_y_row: int = 1,
        color: str = "red",
        linewidth: float = 1.5,
        linestyle: str = "dotted",
        fade: bool = True,
        alpha: float = 0.8,
        label: str = "",
        zorder: int = 4,
    ) -> "CBFAnimator":
        """Add a per-frame prediction line (fading).

        Parameters
        ----------
        source : ``"linear"`` | ``"data"``
            ``"linear"`` computes a constant-velocity prediction from
            position/velocity columns.  ``"data"`` reads from a pre-computed
            list of per-frame trajectories.
        agent_x_idx, agent_y_idx : int
            Position column indices (for ``source="linear"``).
        agent_vx_idx, agent_vy_idx : int
            Velocity column indices (for ``source="linear"``).
        agent_data : np.ndarray, optional
            Data source for position/velocity (defaults to *states*).
        horizon : int
            Number of prediction steps (for ``source="linear"``).
        trajectory_data : list of np.ndarray
            Per-frame trajectory arrays, each with shape ``(>=2, H)``
            (for ``source="data"``).
        traj_x_row, traj_y_row : int
            Row indices for x/y in *trajectory_data* arrays.
        fade : bool
            If True, alpha fades from start to end of the prediction line.
        """
        self._predictions.append(
            {
                "source": source,
                "agent_x_idx": agent_x_idx,
                "agent_y_idx": agent_y_idx,
                "agent_vx_idx": agent_vx_idx,
                "agent_vy_idx": agent_vy_idx,
                "agent_data": agent_data,
                "horizon": horizon,
                "trajectory_data": trajectory_data,
                "traj_x_row": traj_x_row,
                "traj_y_row": traj_y_row,
                "color": color,
                "linewidth": linewidth,
                "linestyle": linestyle,
                "fade": fade,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
            }
        )
        return self

    def show_time(self) -> "CBFAnimator":
        """Enable a time-stamp overlay in the top-left corner."""
        self._show_time = True
        return self

    def on_frame(self, callback: Callable) -> "CBFAnimator":
        """Register a per-frame callback for custom drawing.

        The callback signature is ``(frame_index: int, ax: Axes) -> list``
        and must return a list of modified matplotlib artists (may be empty).

        .. note:: Callbacks are only supported with the ``"matplotlib"``
           backend.  They are silently ignored when using ``"plotly"``.
        """
        self._frame_callbacks.append(callback)
        return self

    def auto_limits(self, margin: float = 1.5) -> "CBFAnimator":
        """Automatically compute axis limits from all data sources.

        Scans states, agent data, goal positions, and sets ``x_lim`` and
        ``y_lim`` to encompass everything with the given *margin*.
        """
        all_x: list = []
        all_y: list = []

        # States used by trajectories
        for spec in self._trajectories:
            src = spec["data"] if spec["data"] is not None else self._states
            all_x.append(src[:, spec["x_idx"]])
            all_y.append(src[:, spec["y_idx"]])

        # Agent positions
        for spec in self._agents:
            src = spec["data"] if spec["data"] is not None else self._states
            all_x.append(src[:, spec["x_idx"]])
            all_y.append(src[:, spec["y_idx"]])

        # Goal positions
        for g in self._goals:
            pos = g["position"]
            all_x.append(np.array([float(pos[0])]))
            all_y.append(np.array([float(pos[1])]))

        # Fallback to raw states if nothing was registered
        if not all_x:
            all_x.append(self._states[:, 0])
            all_y.append(self._states[:, 1])

        xs = np.concatenate(all_x)
        ys = np.concatenate(all_y)
        self._x_lim = (float(xs.min()) - margin, float(xs.max()) + margin)
        self._y_lim = (float(ys.min()) - margin, float(ys.max()) + margin)
        return self

    # ======================================================================
    # Matplotlib backend
    # ======================================================================

    def _build_matplotlib(self) -> Tuple:
        self._fig, self._ax = plt.subplots(figsize=self._config.figsize)
        ax = self._ax

        ax.set_xlim(self._x_lim)
        ax.set_ylim(self._y_lim)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(self._title)
        ax.grid(True, alpha=self._config.grid_alpha)
        if self._aspect is not None:
            ax.set_aspect(self._aspect)

        # --- static elements ---
        for g in self._goals:
            pos = g["position"]
            ax.plot(pos[0], pos[1], "o", color=g["color"], markersize=5, label=g["label"])
            ax.add_patch(
                Circle(
                    (pos[0], pos[1]),
                    g["radius"],
                    color=g["color"],
                    fill=False,
                    linestyle="--",
                    linewidth=1,
                )
            )

        for obs in self._obstacles:
            c = obs["center"]
            if obs["ellipse_radii"] is not None:
                ell = obs["ellipse_radii"]
                ax.add_patch(
                    Ellipse(
                        (c[0], c[1]),
                        width=ell[0] * 2,
                        height=ell[1] * 2,
                        facecolor=obs["color"],
                        alpha=obs["alpha"],
                    )
                )
            elif obs["radius"] is not None:
                ax.add_patch(
                    Circle(
                        (c[0], c[1]),
                        obs["radius"],
                        facecolor=obs["color"],
                        alpha=obs["alpha"],
                        edgecolor=obs["color"],
                        linewidth=2,
                    )
                )

        # --- animated trajectory artists ---
        self._traj_artists = []
        for t in self._trajectories:
            if t["style"] == "scatter":
                (line,) = ax.plot(
                    [],
                    [],
                    linestyle="None",
                    marker=".",
                    markersize=2,
                    alpha=t["alpha"] * _SCATTER_ALPHA_FACTOR,
                    color=t["color"],
                    zorder=t["zorder"] - 1,
                    label=t["label"],
                )
            else:
                (line,) = ax.plot(
                    [],
                    [],
                    color=t["color"],
                    linewidth=t["linewidth"],
                    alpha=t["alpha"],
                    zorder=t["zorder"],
                    label=t["label"],
                )
            self._traj_artists.append(line)

        # --- agent artists ---
        self._agent_artists = []
        for spec in self._agents:
            body = Circle(
                (0, 0), spec["body_radius"],
                color=spec["body_color"], alpha=spec["body_alpha"],
                zorder=spec["zorder"], label=spec["label"],
            )
            ax.add_patch(body)

            zone = None
            if spec["zone_radius"] is not None:
                zone = Circle(
                    (0, 0), spec["zone_radius"],
                    color=spec["zone_color"], alpha=spec["zone_alpha"],
                    zorder=spec["zorder"] - 1,
                )
                ax.add_patch(zone)

            trail_line = None
            if spec["trail"]:
                (trail_line,) = ax.plot(
                    [], [],
                    spec["trail_style"],
                    color=spec["trail_color"],
                    alpha=spec["trail_alpha"],
                    zorder=spec["zorder"] - 1,
                )

            self._agent_artists.append((body, zone, trail_line, spec))

        # --- prediction artists ---
        self._prediction_artists = []
        for spec in self._predictions:
            ls_map = {"dotted": ":", "dashed": "--", "solid": "-"}
            ls = ls_map.get(spec["linestyle"], spec["linestyle"])
            lc = LineCollection(
                [], linewidths=spec["linewidth"],
                linestyles=ls, colors=spec["color"],
                zorder=spec["zorder"],
            )
            if spec["label"]:
                lc.set_label(spec["label"])
            ax.add_collection(lc)
            self._prediction_artists.append((lc, spec))

        # --- time overlay ---
        self._time_text = None
        if self._show_time:
            self._time_text = ax.text(
                0.02,
                0.95,
                "",
                transform=ax.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
            )

        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        return self._fig, self._ax

    def _init_func(self):
        artists: list = []

        for line in self._traj_artists:
            line.set_data([], [])
        artists.extend(self._traj_artists)

        for body, zone, trail_line, spec in self._agent_artists:
            src = spec["data"] if spec["data"] is not None else self._states
            body.center = (float(src[0, spec["x_idx"]]), float(src[0, spec["y_idx"]]))
            artists.append(body)
            if zone is not None:
                zone.center = body.center
                artists.append(zone)
            if trail_line is not None:
                trail_line.set_data([], [])
                artists.append(trail_line)

        for lc, _spec in self._prediction_artists:
            lc.set_segments([])
            artists.append(lc)

        if self._time_text is not None:
            self._time_text.set_text("")
            artists.append(self._time_text)

        return artists

    def _update_func(self, frame):
        artists: list = []

        # Trajectories
        for spec, line in zip(self._trajectories, self._traj_artists):
            data = spec["data"] if spec["data"] is not None else self._states
            line.set_data(data[:frame, spec["x_idx"]], data[:frame, spec["y_idx"]])
            artists.append(line)

        # Agents
        for body, zone, trail_line, spec in self._agent_artists:
            src = spec["data"] if spec["data"] is not None else self._states
            pos = (float(src[frame, spec["x_idx"]]), float(src[frame, spec["y_idx"]]))
            body.center = pos
            artists.append(body)
            if zone is not None:
                zone.center = pos
                artists.append(zone)
            if trail_line is not None:
                trail_line.set_data(src[:frame, spec["x_idx"]], src[:frame, spec["y_idx"]])
                artists.append(trail_line)

        # Predictions
        for lc, spec in self._prediction_artists:
            px, py = self._compute_prediction(spec, frame)
            if len(px) >= 2:
                segs, alphas = _get_fading_segments(px, py)
                lc.set_segments(segs)
                if spec["fade"]:
                    lc.set_alpha(alphas)
                else:
                    lc.set_alpha(spec["alpha"])
            else:
                lc.set_segments([])
            artists.append(lc)

        # Time overlay
        if self._time_text is not None:
            self._time_text.set_text(f"Time: {frame * self._dt:.1f}s")
            artists.append(self._time_text)

        # Custom callbacks
        for cb in self._frame_callbacks:
            extra = cb(frame, self._ax)
            if extra:
                artists.extend(extra)

        return artists

    def _compute_prediction(self, spec: dict, frame: int):
        """Compute prediction x/y arrays for a given frame."""
        if spec["source"] == "linear":
            src = spec["agent_data"] if spec["agent_data"] is not None else self._states
            px_pos = float(src[frame, spec["agent_x_idx"]])
            py_pos = float(src[frame, spec["agent_y_idx"]])
            vx = float(src[frame, spec["agent_vx_idx"]])
            vy = float(src[frame, spec["agent_vy_idx"]])
            h = spec["horizon"]
            px = [px_pos + vx * self._dt * k for k in range(h)]
            py = [py_pos + vy * self._dt * k for k in range(h)]
            return px, py
        elif spec["source"] == "data":
            traj_data = spec["trajectory_data"]
            if traj_data is not None and frame < len(traj_data):
                traj = np.asarray(traj_data[frame])
                if traj.ndim >= 2 and traj.shape[0] > max(spec["traj_x_row"], spec["traj_y_row"]):
                    return traj[spec["traj_x_row"], :].tolist(), traj[spec["traj_y_row"], :].tolist()
            return [], []
        return [], []

    def _animate_matplotlib(self):
        if self._fig is None:
            self._build_matplotlib()

        # Real-time: interval in ms matches the simulation timestep
        interval_ms = self._dt * 1000

        self._anim = mpl_animation.FuncAnimation(
            self._fig,
            self._update_func,
            frames=len(self._states),
            init_func=self._init_func,
            blit=self._config.blit,
            interval=interval_ms,
        )
        return self._anim

    def _save_matplotlib(self, path: str, config: Optional[AnimationConfig] = None) -> str:
        if self._anim is None:
            self._animate_matplotlib()
        # Use real-time fps derived from dt
        save_cfg = AnimationConfig(**(config or self._config).__dict__)
        save_cfg.fps = int(round(1.0 / self._dt))
        return save_animation(self._anim, path, save_cfg)

    def _show_matplotlib(self):
        if self._anim is None:
            self._animate_matplotlib()
        plt.show()

    # ======================================================================
    # Plotly backend
    # ======================================================================

    def _build_plotly(self):
        """Build an animated Plotly figure from the declared elements."""
        cfg = self._config
        n_total = len(self._states)
        frame_indices, frame_duration_ms = _compute_plotly_frame_step(
            self._dt, n_total, max_frames=cfg.plotly_max_frames,
        )

        # --- static layout shapes (goals + obstacles) ---
        static_shapes: list = []
        for g in self._goals:
            pos = g["position"]
            css = _to_css_color(g["color"])
            static_shapes.append(
                dict(
                    type="circle",
                    xref="x", yref="y",
                    x0=float(pos[0]) - g["radius"],
                    y0=float(pos[1]) - g["radius"],
                    x1=float(pos[0]) + g["radius"],
                    y1=float(pos[1]) + g["radius"],
                    line=dict(color=css, dash="dash", width=1),
                    fillcolor="rgba(0,0,0,0)",
                )
            )

        for obs in self._obstacles:
            c = obs["center"]
            css = _to_css_color(obs["color"])
            opacity = obs["alpha"]
            if obs["ellipse_radii"] is not None:
                rx, ry = obs["ellipse_radii"]
            elif obs["radius"] is not None:
                rx = ry = obs["radius"]
            else:
                continue
            path_str = _ellipse_svg_path(float(c[0]), float(c[1]), float(rx), float(ry))
            static_shapes.append(
                dict(
                    type="path",
                    path=path_str,
                    xref="x", yref="y",
                    fillcolor=css,
                    opacity=opacity,
                    line=dict(color=css, width=1),
                )
            )

        # --- base traces ---
        base_traces: list = []
        # Track how many traces each element type contributes
        # so we can build the per-frame `traces` index list correctly.

        # 1) Goal markers
        for g in self._goals:
            pos = g["position"]
            css = _to_css_color(g["color"])
            base_traces.append(
                go.Scatter(
                    x=[float(pos[0])],
                    y=[float(pos[1])],
                    mode="markers",
                    marker=dict(size=8, color=css, symbol="circle"),
                    name=g["label"],
                    showlegend=True,
                )
            )

        # 2) Trajectory traces
        for spec in self._trajectories:
            css = _to_css_color(spec["color"])
            mode = "markers" if spec["style"] == "scatter" else "lines"
            marker_opts = (
                dict(size=3, color=css, opacity=spec["alpha"] * _SCATTER_ALPHA_FACTOR)
                if spec["style"] == "scatter"
                else dict()
            )
            line_opts = (
                dict(color=css, width=spec["linewidth"])
                if spec["style"] != "scatter"
                else dict()
            )
            base_traces.append(
                go.Scatter(
                    x=[], y=[],
                    mode=mode,
                    name=spec["label"],
                    line=line_opts if line_opts else None,
                    marker=marker_opts if marker_opts else None,
                    opacity=spec["alpha"] if spec["style"] != "scatter" else 1.0,
                )
            )

        # 3) Agent traces: body marker + trail line per agent
        for spec in self._agents:
            css = _to_css_color(spec["body_color"])
            # Body marker
            base_traces.append(
                go.Scatter(
                    x=[], y=[],
                    mode="markers",
                    marker=dict(size=10, color=css, opacity=spec["body_alpha"]),
                    name=spec["label"],
                    showlegend=True,
                )
            )
            # Trail line
            trail_css = _to_css_color(spec["trail_color"])
            dash_map = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
            dash = dash_map.get(spec["trail_style"], "solid")
            base_traces.append(
                go.Scatter(
                    x=[], y=[],
                    mode="lines",
                    line=dict(color=trail_css, width=1.5, dash=dash),
                    opacity=spec["trail_alpha"],
                    name=spec["label"] + " trail",
                    showlegend=False,
                )
            )

        # 4) Prediction traces: _PLOTLY_FADE_BUCKETS sub-traces per prediction
        for spec in self._predictions:
            css = _to_css_color(spec["color"])
            dash_map = {"dotted": "dot", "dashed": "dash", "solid": "solid"}
            dash = dash_map.get(spec["linestyle"], "solid")
            for bucket in range(_PLOTLY_FADE_BUCKETS):
                opacity = spec["alpha"] * (1.0 - bucket / _PLOTLY_FADE_BUCKETS)
                base_traces.append(
                    go.Scatter(
                        x=[], y=[],
                        mode="lines",
                        line=dict(color=css, width=spec["linewidth"], dash=dash),
                        opacity=opacity,
                        name=spec["label"] if (bucket == 0 and spec["label"]) else "",
                        showlegend=(bucket == 0 and bool(spec["label"])),
                    )
                )

        n_traces = len(base_traces)

        # --- animation frames ---
        has_agents = len(self._agents) > 0
        frames: list = []
        for fi in frame_indices:
            t = fi * self._dt
            trace_updates: list = []

            # 1) Goal markers (unchanged)
            for g in self._goals:
                pos = g["position"]
                trace_updates.append(
                    go.Scatter(x=[float(pos[0])], y=[float(pos[1])])
                )

            # 2) Trajectory traces
            for spec in self._trajectories:
                src = spec["data"] if spec["data"] is not None else self._states
                trace_updates.append(
                    go.Scatter(
                        x=src[:fi, spec["x_idx"]].tolist(),
                        y=src[:fi, spec["y_idx"]].tolist(),
                    )
                )

            # 3) Agent traces (body marker + trail)
            for spec in self._agents:
                src = spec["data"] if spec["data"] is not None else self._states
                px = float(src[fi, spec["x_idx"]])
                py = float(src[fi, spec["y_idx"]])
                # Body
                trace_updates.append(go.Scatter(x=[px], y=[py]))
                # Trail
                trace_updates.append(
                    go.Scatter(
                        x=src[:fi, spec["x_idx"]].tolist(),
                        y=src[:fi, spec["y_idx"]].tolist(),
                    )
                )

            # 4) Prediction traces (bucketed fading)
            for spec in self._predictions:
                px, py = self._compute_prediction(spec, fi)
                n_pts = len(px)
                for bucket in range(_PLOTLY_FADE_BUCKETS):
                    if n_pts >= 2:
                        start = bucket * n_pts // _PLOTLY_FADE_BUCKETS
                        end = (bucket + 1) * n_pts // _PLOTLY_FADE_BUCKETS + 1
                        end = min(end, n_pts)
                        trace_updates.append(
                            go.Scatter(
                                x=list(px[start:end]),
                                y=list(py[start:end]),
                            )
                        )
                    else:
                        trace_updates.append(go.Scatter(x=[], y=[]))

            # --- per-frame layout (time annotation + agent shapes) ---
            layout_update = {}

            if self._show_time:
                layout_update["annotations"] = [
                    dict(
                        x=0.02, y=0.98,
                        xref="paper", yref="paper",
                        text=f"Time: {t:.1f}s",
                        showarrow=False,
                        font=dict(size=13),
                        bgcolor="white",
                        opacity=0.8,
                        bordercolor="gray",
                        borderwidth=1,
                    )
                ]

            # Agent body/zone as moving shapes
            if has_agents:
                frame_shapes = list(static_shapes)  # copy static shapes
                for spec in self._agents:
                    src = spec["data"] if spec["data"] is not None else self._states
                    ax_pos = float(src[fi, spec["x_idx"]])
                    ay_pos = float(src[fi, spec["y_idx"]])
                    # Body circle
                    frame_shapes.append(
                        _circle_shape(
                            ax_pos, ay_pos, spec["body_radius"],
                            spec["body_color"], alpha=spec["body_alpha"],
                        )
                    )
                    # Safety zone
                    if spec["zone_radius"] is not None:
                        frame_shapes.append(
                            _circle_shape(
                                ax_pos, ay_pos, spec["zone_radius"],
                                spec["zone_color"], alpha=spec["zone_alpha"],
                            )
                        )
                layout_update["shapes"] = frame_shapes

            frames.append(
                go.Frame(
                    data=trace_updates,
                    traces=list(range(n_traces)),
                    name=f"{t:.2f}s",
                    layout=layout_update if layout_update else None,
                )
            )

        # --- assemble figure ---
        plot_size = 600
        menus, sliders = _plotly_animation_controls(frames, frame_duration_ms)

        fig = go.Figure(
            data=base_traces,
            frames=frames,
            layout=go.Layout(
                title=dict(
                    text=self._title,
                    x=0.5, xanchor="center",
                    font=dict(size=16),
                ),
                xaxis=dict(
                    title="x [m]",
                    range=list(self._x_lim),
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.1)",
                    constrain="domain",
                ),
                yaxis=dict(
                    title="y [m]",
                    range=list(self._y_lim),
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.1)",
                    **(dict(scaleanchor="x", scaleratio=1) if self._aspect == "equal" else {}),
                    constrain="domain",
                ),
                shapes=static_shapes,
                width=plot_size + 90,
                height=plot_size + 160,
                margin=dict(l=60, r=30, t=60, b=100),
                updatemenus=menus,
                sliders=sliders,
                legend=dict(
                    x=1.0, y=1.0,
                    xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                ),
                template="plotly_white",
            ),
        )

        self._fig = fig
        return fig

    def _save_plotly(self, path: str) -> str:
        if self._fig is None:
            self._build_plotly()

        output_path = Path(path).with_suffix(".html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fig.write_html(str(output_path), auto_open=False)
        abs_path = str(output_path.resolve())
        print(f"\nInteractive animation saved to: file://{abs_path}")
        return abs_path

    def _show_plotly(self):
        if self._fig is None:
            self._build_plotly()
        self._fig.show()

    # ======================================================================
    # Public API -- dispatches to the active backend
    # ======================================================================

    def build(self):
        """Create the figure and all static / dynamic artists.

        Returns ``(fig, ax)`` for the matplotlib backend, or the Plotly
        ``Figure`` for the plotly backend.
        """
        if self._backend == "plotly":
            return self._build_plotly()
        return self._build_matplotlib()

    def animate(self):
        """Build (if needed) and create the animation object.

        Returns a matplotlib ``FuncAnimation`` or a Plotly ``Figure``
        (which already contains the animation frames).
        """
        if self._backend == "plotly":
            if self._fig is None:
                self._build_plotly()
            return self._fig
        return self._animate_matplotlib()

    def save(self, path: str, config: Optional[AnimationConfig] = None) -> str:
        """Animate (if needed) and save.

        * matplotlib: saves MP4 (ffmpeg) or GIF (pillow fallback).
        * plotly: saves an interactive ``.html`` file.

        Returns the absolute path of the saved file.
        """
        if self._backend == "plotly":
            return self._save_plotly(path)
        return self._save_matplotlib(path, config)

    def show(self):
        """Animate (if needed) and display interactively.

        * matplotlib: opens a matplotlib window.
        * plotly: opens the default web browser.
        """
        if self._backend == "plotly":
            return self._show_plotly()
        return self._show_matplotlib()

    # -- properties ---------------------------------------------------------

    @property
    def fig(self):
        """The matplotlib Figure or Plotly Figure (available after :meth:`build`)."""
        return self._fig

    @property
    def ax(self):
        """The matplotlib Axes (available after :meth:`build`).  *None* for Plotly."""
        return self._ax

    @property
    def animation(self):
        """The FuncAnimation or Plotly Figure (available after :meth:`animate`)."""
        if self._backend == "plotly":
            return self._fig
        return self._anim
