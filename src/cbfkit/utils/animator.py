"""Reusable animation framework for CBFKit simulations.

Provides :class:`CBFAnimator` for building 2D trajectory animations with a
declarative API, :class:`AnimationConfig` for centralised defaults, and
:func:`save_animation` for robust file output with format fallback.

Supports two backends:

* ``"matplotlib"`` (default) — generates MP4 / GIF animations via
  ``FuncAnimation``.
* ``"plotly"`` — generates interactive HTML files with play/pause controls
  and a timeline slider.  Requires ``pip install cbfkit[plotly]``.

Example
-------
>>> from cbfkit.utils.animator import CBFAnimator
>>> anim = CBFAnimator(states, dt=0.1, x_lim=(-4, 4), y_lim=(-4, 4))
>>> anim.add_goal(desired_state[:2], radius=0.25)
>>> anim.add_obstacles(obstacle_centers, ellipsoid_radii=ellipsoid_radii)
>>> anim.add_trajectory(x_idx=0, y_idx=1, label="Trajectory")
>>> anim.save("results/animation.mp4")

Interactive Plotly version:

>>> anim = CBFAnimator(states, dt=0.1, backend="plotly")
>>> anim.add_goal(desired_state[:2], radius=0.25)
>>> anim.add_trajectory(x_idx=0, y_idx=1, label="Trajectory")
>>> anim.save("results/animation.html")  # interactive HTML
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


# Matplotlib short color names / tab: palette → CSS hex for Plotly
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
        ``"matplotlib"`` (default) or ``"plotly"``.
    config : AnimationConfig, optional
        Animation parameters.  Uses :data:`DEFAULT_CONFIG` when *None*.

    Example
    -------
    >>> anim = CBFAnimator(states, dt=0.1, x_lim=(-4, 4), y_lim=(-4, 4))
    >>> anim.add_goal([1, 2], radius=0.25)
    >>> anim.add_trajectory(x_idx=0, y_idx=1, label="Robot")
    >>> anim.save("output.mp4")

    Interactive Plotly output:

    >>> anim = CBFAnimator(states, dt=0.1, backend="plotly")
    >>> anim.add_trajectory(x_idx=0, y_idx=1)
    >>> anim.show()   # opens in browser
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
        self._show_time: bool = False
        self._frame_callbacks: List[Callable] = []

        # Matplotlib objects (created by build / animate)
        self._fig = None
        self._ax = None
        self._anim = None
        self._traj_artists: List = []
        self._time_text = None

        # Plotly figure
        self._plotly_fig = None

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
                    alpha=t["alpha"] * 0.55,
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
        for line in self._traj_artists:
            line.set_data([], [])
        artists = list(self._traj_artists)
        if self._time_text is not None:
            self._time_text.set_text("")
            artists.append(self._time_text)
        return artists

    def _update_func(self, frame):
        artists: list = []
        for spec, line in zip(self._trajectories, self._traj_artists):
            data = spec["data"] if spec["data"] is not None else self._states
            line.set_data(data[:frame, spec["x_idx"]], data[:frame, spec["y_idx"]])
            artists.append(line)

        if self._time_text is not None:
            self._time_text.set_text(f"Time: {frame * self._dt:.1f}s")
            artists.append(self._time_text)

        for cb in self._frame_callbacks:
            extra = cb(frame, self._ax)
            if extra:
                artists.extend(extra)

        return artists

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
        step = max(1, n_total // cfg.plotly_max_frames)
        frame_indices = list(range(0, n_total, step))
        if frame_indices[-1] != n_total - 1:
            frame_indices.append(n_total - 1)

        # Real-time: each frame covers step * dt seconds of simulation
        frame_duration_ms = self._dt * step * 1000

        # --- layout shapes for static elements ---
        shapes: list = []
        for g in self._goals:
            pos = g["position"]
            css = _to_css_color(g["color"])
            shapes.append(
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
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
            # SVG path for an axis-aligned ellipse
            path_str = _ellipse_svg_path(float(c[0]), float(c[1]), float(rx), float(ry))
            shapes.append(
                dict(
                    type="path",
                    path=path_str,
                    xref="x",
                    yref="y",
                    fillcolor=css,
                    opacity=opacity,
                    line=dict(color=css, width=1),
                )
            )

        # --- base traces (initial frame = empty) ---
        base_traces: list = []

        # Goal markers
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

        # Trajectory traces (one per add_trajectory call)
        for spec in self._trajectories:
            css = _to_css_color(spec["color"])
            mode = "markers" if spec["style"] == "scatter" else "lines"
            marker_opts = (
                dict(size=3, color=css, opacity=spec["alpha"] * 0.55)
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
                    x=[],
                    y=[],
                    mode=mode,
                    name=spec["label"],
                    line=line_opts if line_opts else None,
                    marker=marker_opts if marker_opts else None,
                    opacity=spec["alpha"] if spec["style"] != "scatter" else 1.0,
                )
            )

        # --- animation frames ---
        n_goals = len(self._goals)
        frames: list = []
        for fi in frame_indices:
            t = fi * self._dt
            trace_updates: list = []
            # Goal markers don't change — emit them unchanged
            for g in self._goals:
                pos = g["position"]
                trace_updates.append(go.Scatter(x=[float(pos[0])], y=[float(pos[1])]))

            # Trajectory traces
            for spec in self._trajectories:
                src = spec["data"] if spec["data"] is not None else self._states
                trace_updates.append(
                    go.Scatter(
                        x=src[:fi, spec["x_idx"]].tolist(),
                        y=src[:fi, spec["y_idx"]].tolist(),
                    )
                )

            # Time annotation (per-frame layout update)
            layout_update = {}
            if self._show_time:
                layout_update["annotations"] = [
                    dict(
                        x=0.02,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=f"Time: {t:.1f}s",
                        showarrow=False,
                        font=dict(size=13),
                        bgcolor="white",
                        opacity=0.8,
                        bordercolor="gray",
                        borderwidth=1,
                    )
                ]

            frames.append(
                go.Frame(
                    data=trace_updates,
                    traces=list(range(len(trace_updates))),
                    name=f"{t:.2f}s",
                    layout=layout_update if layout_update else None,
                )
            )

        # --- assemble figure ---
        # Force square plot area with constrained axes
        plot_size = 600

        fig = go.Figure(
            data=base_traces,
            frames=frames,
            layout=go.Layout(
                title=dict(
                    text=self._title,
                    x=0.5,
                    xanchor="center",
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
                    scaleanchor="x",
                    scaleratio=1,
                    constrain="domain",
                ),
                shapes=shapes,
                width=plot_size + 90,
                height=plot_size + 160,
                margin=dict(l=60, r=30, t=60, b=100),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        y=-0.32,
                        x=0.5,
                        xanchor="center",
                        yanchor="top",
                        direction="left",
                        pad=dict(t=0, r=10),
                        buttons=[
                            dict(
                                label="\u25b6 Play",
                                method="animate",
                                args=[
                                    None,
                                    dict(
                                        frame=dict(
                                            duration=frame_duration_ms,
                                            redraw=True,
                                        ),
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
                ],
                sliders=[
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
                        x=0.1,
                        len=0.8,
                        y=-0.08,
                        yanchor="top",
                        currentvalue=dict(
                            prefix="Time: ",
                            visible=True,
                            xanchor="center",
                        ),
                        transition=dict(duration=0),
                    )
                ],
                legend=dict(
                    x=1.0,
                    y=1.0,
                    xanchor="right",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                ),
                template="plotly_white",
            ),
        )

        self._plotly_fig = fig
        return fig

    def _save_plotly(self, path: str) -> str:
        if self._plotly_fig is None:
            self._build_plotly()

        output_path = Path(path).with_suffix(".html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._plotly_fig.write_html(str(output_path), auto_open=False)
        abs_path = str(output_path.resolve())
        print(f"\nInteractive animation saved to: file://{abs_path}")
        return abs_path

    def _show_plotly(self):
        if self._plotly_fig is None:
            self._build_plotly()
        self._plotly_fig.show()

    # ======================================================================
    # Public API — dispatches to the active backend
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
            if self._plotly_fig is None:
                self._build_plotly()
            return self._plotly_fig
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
        if self._backend == "plotly":
            return self._plotly_fig
        return self._fig

    @property
    def ax(self):
        """The matplotlib Axes (available after :meth:`build`).  *None* for Plotly."""
        return self._ax

    @property
    def animation(self):
        """The FuncAnimation or Plotly Figure (available after :meth:`animate`)."""
        if self._backend == "plotly":
            return self._plotly_fig
        return self._anim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ellipse_svg_path(cx: float, cy: float, rx: float, ry: float) -> str:
    """Return an SVG path string for an axis-aligned ellipse.

    Uses four cubic Bezier segments (the standard circle-to-Bezier
    approximation scaled to ``rx``, ``ry``).  Plotly layout shapes do
    **not** support SVG arc (``A``) commands, so Bezier curves are required.
    """
    # Kappa: optimal handle length for a quarter-circle Bezier approximation
    k = 0.5522847498
    kx, ky = rx * k, ry * k
    return (
        f"M {cx},{cy - ry} "
        f"C {cx + kx},{cy - ry} {cx + rx},{cy - ky} {cx + rx},{cy} "
        f"C {cx + rx},{cy + ky} {cx + kx},{cy + ry} {cx},{cy + ry} "
        f"C {cx - kx},{cy + ry} {cx - rx},{cy + ky} {cx - rx},{cy} "
        f"C {cx - rx},{cy - ky} {cx - kx},{cy - ry} {cx},{cy - ry} Z"
    )
