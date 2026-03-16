"""Core CBFAnimator class with declarative element API."""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .config import AnimationConfig
from .deps import _require_manim, _require_matplotlib, _require_plotly
from .matplotlib_backend import _MatplotlibMixin
from .plotly_backend import _PlotlyMixin


class CBFAnimator(_MatplotlibMixin, _PlotlyMixin):
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
        if backend not in ("matplotlib", "plotly") and not backend.startswith("manim"):
            raise ValueError(
                f"Unknown backend {backend!r}. "
                "Use 'matplotlib', 'plotly', or 'manim' / 'manim-<quality>'."
            )

        self._backend = backend
        if backend == "matplotlib":
            _require_matplotlib()
        elif backend.startswith("manim"):
            _require_manim()
            raise NotImplementedError("Manim 2D backend not yet implemented. "
                                      "Use visualize_3d_multi_robot(backend='manim') for 3D scenes.")
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
            If provided, draws a translucent circle around the body
            representing a safety / social zone.
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

    # -- prediction computation (shared by both backends) -------------------

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
