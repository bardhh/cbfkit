"""Reusable animation framework for CBFKit simulations.

Provides :class:`CBFAnimator` for building 2D trajectory animations with a
declarative API, :class:`AnimationConfig` for centralised defaults, and
:func:`save_animation` for robust file output with format fallback.

Example
-------
>>> from cbfkit.utils.animator import CBFAnimator
>>> anim = CBFAnimator(states, dt=0.1, x_lim=(-4, 4), y_lim=(-4, 4))
>>> anim.add_goal(desired_state[:2], radius=0.25)
>>> anim.add_obstacles(obstacle_centers, ellipsoid_radii=ellipsoid_radii)
>>> anim.add_trajectory(x_idx=0, y_idx=1, label="Trajectory")
>>> anim.save("results/animation.mp4")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _require_matplotlib():
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "Optional dependency 'matplotlib' not found. "
            "Please install cbfkit[vis] to use visualization features."
        )


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
    """

    fps: int = 20
    dpi: int = 100
    interval: int = 50
    bitrate: int = 1800
    figsize: Tuple[float, float] = (10, 8)
    grid_alpha: float = 0.3
    blit: bool = True


DEFAULT_CONFIG = AnimationConfig()
"""Module-level default :class:`AnimationConfig` instance."""


# ---------------------------------------------------------------------------
# Standalone save helper
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
    config : AnimationConfig, optional
        Animation parameters.  Uses :data:`DEFAULT_CONFIG` when *None*.

    Example
    -------
    >>> anim = CBFAnimator(states, dt=0.1, x_lim=(-4, 4), y_lim=(-4, 4))
    >>> anim.add_goal([1, 2], radius=0.25)
    >>> anim.add_trajectory(x_idx=0, y_idx=1, label="Robot")
    >>> anim.save("output.mp4")
    """

    def __init__(
        self,
        states: np.ndarray,
        dt: float = 0.1,
        x_lim: Tuple[float, float] = (-4, 4),
        y_lim: Tuple[float, float] = (-4, 4),
        title: str = "System Behavior",
        aspect: Optional[str] = None,
        config: Optional[AnimationConfig] = None,
    ):
        _require_matplotlib()
        self._states = states
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
        self._fig: Optional["plt.Figure"] = None
        self._ax: Optional["plt.Axes"] = None
        self._anim: Optional["mpl_animation.FuncAnimation"] = None
        self._traj_artists: List = []
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

    def show_time(self) -> "CBFAnimator":
        """Enable a time-stamp overlay in the top-left corner."""
        self._show_time = True
        return self

    def on_frame(self, callback: Callable) -> "CBFAnimator":
        """Register a per-frame callback for custom drawing.

        The callback signature is ``(frame_index: int, ax: Axes) -> list``
        and must return a list of modified matplotlib artists (may be empty).
        """
        self._frame_callbacks.append(callback)
        return self

    # -- build & animate ----------------------------------------------------

    def build(self) -> Tuple["plt.Figure", "plt.Axes"]:
        """Create the figure, axes, and all static / dynamic artists.

        This is called automatically by :meth:`animate` if the figure has not
        been built yet, but you can call it early to customise the axes before
        animation starts.
        """
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

    def animate(self) -> "mpl_animation.FuncAnimation":
        """Build (if needed) and create the :class:`FuncAnimation`."""
        if self._fig is None:
            self.build()

        self._anim = mpl_animation.FuncAnimation(
            self._fig,
            self._update_func,
            frames=len(self._states),
            init_func=self._init_func,
            blit=self._config.blit,
            interval=self._config.interval,
        )
        return self._anim

    def save(self, path: str, config: Optional[AnimationConfig] = None) -> str:
        """Animate (if needed) and save to *path* with format fallback.

        Returns the absolute path the file was saved to.
        """
        if self._anim is None:
            self.animate()
        return save_animation(self._anim, path, config or self._config)

    def show(self):
        """Animate (if needed) and display interactively."""
        if self._anim is None:
            self.animate()
        plt.show()

    # -- properties ---------------------------------------------------------

    @property
    def fig(self) -> Optional["plt.Figure"]:
        """The matplotlib Figure (available after :meth:`build`)."""
        return self._fig

    @property
    def ax(self) -> Optional["plt.Axes"]:
        """The matplotlib Axes (available after :meth:`build`)."""
        return self._ax

    @property
    def animation(self) -> Optional["mpl_animation.FuncAnimation"]:
        """The FuncAnimation (available after :meth:`animate`)."""
        return self._anim
