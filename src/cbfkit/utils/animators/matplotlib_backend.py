"""Matplotlib backend mixin for CBFAnimator."""

from typing import Optional

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse

from .config import AnimationConfig
from .helpers import _SCATTER_ALPHA_FACTOR, _get_fading_segments, save_animation


class _MatplotlibMixin:
    """Matplotlib-specific build / animate / save / show methods.

    Mixed into :class:`~cbfkit.utils.animators.animator.CBFAnimator`.
    Expects the host class to provide ``_states``, ``_dt``, ``_config``,
    ``_goals``, ``_obstacles``, ``_trajectories``, ``_agents``,
    ``_predictions``, ``_show_time``, ``_frame_callbacks``, ``_x_lim``,
    ``_y_lim``, ``_title``, ``_aspect``, and ``_compute_prediction``.
    """

    def _build_matplotlib(self):
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
