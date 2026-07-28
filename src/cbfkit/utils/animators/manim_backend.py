"""Manim 2D backend mixin for CBFAnimator.

Renders the declarative element API (goals, obstacles, trajectories, agents,
predictions, time overlay) as a high-quality Manim video.  Mirrors the
architecture of :mod:`cbfkit.utils.visualizations.manim_3d_multi_robot`:
data is injected into a Scene subclass via class attributes, then rendered.

Usage (standalone test with synthetic data):
    manim -pql src/cbfkit/utils/animators/manim_backend.py CBFAnimator2DScene

Usage (from library):
    from cbfkit.utils.animator import CBFAnimator
    CBFAnimator(states, backend="manim-medium").add_agent(0, 1).save("out.mp4")
"""

from __future__ import annotations

import os
import shutil

import numpy as np

from .deps import _require_manim

try:
    from manim import (
        UL,
        UP,
        Circle,
        DashedLine,
        DashedVMobject,
        Dot,
        Ellipse,
        Line,
        Scene,
        Text,
        TracedPath,  # noqa: F401  (re-exported for parity with the 3D module)
        ValueTracker,
        VGroup,
        rate_functions,
        tempconfig,
    )

    _MANIM_AVAILABLE = True
except ImportError:
    _MANIM_AVAILABLE = False
    # Fallback base class so this module still imports when Manim is absent;
    # actual rendering is gated by _require_manim() in _ManimMixin methods.
    Scene = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Colour handling: translate matplotlib-style colour strings to hex
# ---------------------------------------------------------------------------
_TAB10_HEX = {
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

_NAMED_HEX = {
    "r": "#d62728",
    "red": "#d62728",
    "g": "#2ca02c",
    "green": "#2ca02c",
    "b": "#1f77b4",
    "blue": "#1f77b4",
    "k": "#000000",
    "black": "#000000",
    "w": "#ffffff",
    "white": "#ffffff",
    "y": "#bcbd22",
    "yellow": "#bcbd22",
    "c": "#17becf",
    "cyan": "#17becf",
    "m": "#e377c2",
    "magenta": "#e377c2",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "grey": "#7f7f7f",
}


def _to_hex(color: str) -> str:
    """Best-effort conversion of a matplotlib-style colour string to hex."""
    if not isinstance(color, str) or color.startswith("#"):
        return color
    if color in _TAB10_HEX:
        return _TAB10_HEX[color]
    return _NAMED_HEX.get(color.lower(), color)


# ---------------------------------------------------------------------------
# Core Manim Scene
# ---------------------------------------------------------------------------
class CBFAnimator2DScene(Scene):
    """Render a 2D CBFAnimator element set as a Manim animation.

    Pass data via class attributes before calling ``scene.render()``, or let
    the scene generate synthetic demo data when run standalone.
    """

    # --- class-level defaults (overridden by _ManimMixin._build_manim) -----
    states: np.ndarray | None = None
    dt: float = 0.1
    x_lim: tuple[float, float] = (-4, 4)
    y_lim: tuple[float, float] = (-4, 4)
    title: str = "System Behavior"
    aspect: str | None = None
    goals: list[dict] = []
    obstacles: list[dict] = []
    trajectories: list[dict] = []
    agents: list[dict] = []
    predictions: list[dict] = []
    show_time_overlay: bool = False
    # Bound CBFAnimator._compute_prediction (set by _build_manim)
    compute_prediction = None

    # -----------------------------------------------------------------------
    def construct(self):
        if self.states is None:
            self._synthetic_demo()

        states = np.asarray(self.states)
        n_frames = len(states)

        # White canvas so matplotlib-convention colours (black obstacles,
        # dark trajectories) stay visible, matching the other backends.
        self.camera.background_color = "#ffffff"

        axes, unit_x, unit_y = self._build_axes()
        self.add(axes)

        if self.title:
            self.add(Text(self.title, font_size=28, color="#000000").to_edge(UP, buff=0.2))

        def c2p(x: float, y: float):
            return axes.c2p(float(x), float(y))

        # --- static goals ---------------------------------------------------
        for g in self.goals:
            pos, color = g["position"], _to_hex(g["color"])
            dot = Dot(c2p(pos[0], pos[1]), radius=0.06, color=color)
            ring = Circle(radius=g["radius"] * unit_x, color=color, stroke_width=2)
            ring.move_to(c2p(pos[0], pos[1]))
            self.add(dot, DashedVMobject(ring, num_dashes=24))

        # --- static obstacles -------------------------------------------------
        for obs in self.obstacles:
            c, color = obs["center"], _to_hex(obs["color"])
            if obs["ellipse_radii"] is not None:
                rx, ry = obs["ellipse_radii"]
                mob = Ellipse(width=2 * rx * unit_x, height=2 * ry * unit_y, color=color)
            elif obs["radius"] is not None:
                mob = Circle(radius=obs["radius"] * unit_x, color=color)
            else:
                continue
            mob.set_fill(color, opacity=obs["alpha"]).set_stroke(color, width=2)
            mob.move_to(c2p(c[0], c[1]))
            self.add(mob)

        # --- progress tracker -------------------------------------------------
        progress = ValueTracker(0)

        def current_frame() -> int:
            return min(int(progress.get_value()), n_frames - 1)

        def _reveal_updater(segments_or_dots):
            """Reveal pre-built per-frame mobjects up to the current frame."""

            def updater(_mob):
                frame = current_frame()
                for k, seg in enumerate(segments_or_dots):
                    seg.set_opacity(seg.target_opacity if k < frame else 0.0)

            return updater

        def _build_reveal_group(xs, ys, color, *, style, stroke_width, opacity, dashed=False):
            """Pre-build per-frame segments (line) or dots (scatter), initially hidden."""
            group = VGroup()
            if style == "scatter":
                for k in range(len(xs)):
                    d = Dot(c2p(xs[k], ys[k]), radius=0.03, color=color)
                    d.target_opacity = opacity
                    d.set_opacity(0)
                    group.add(d)
            else:
                line_cls = DashedLine if dashed else Line
                for k in range(len(xs) - 1):
                    seg = line_cls(
                        c2p(xs[k], ys[k]),
                        c2p(xs[k + 1], ys[k + 1]),
                        stroke_width=stroke_width,
                        color=color,
                    )
                    seg.target_opacity = opacity
                    seg.set_opacity(0)
                    group.add(seg)
            group.add_updater(_reveal_updater(list(group)))
            return group

        # --- animated trajectories --------------------------------------------
        for spec in self.trajectories:
            data = spec["data"] if spec["data"] is not None else states
            group = _build_reveal_group(
                data[:, spec["x_idx"]],
                data[:, spec["y_idx"]],
                _to_hex(spec["color"]),
                style=spec["style"],
                stroke_width=2 * spec["linewidth"],
                opacity=spec["alpha"],
            )
            self.add(group)

        # --- agents (body + optional zone + optional trail) --------------------
        for spec in self.agents:
            src = spec["data"] if spec["data"] is not None else states
            x_idx, y_idx = spec["x_idx"], spec["y_idx"]
            color = _to_hex(spec["body_color"])
            start = c2p(src[0, x_idx], src[0, y_idx])

            body = Circle(radius=spec["body_radius"] * unit_x, color=color)
            body.set_fill(color, opacity=spec["body_alpha"]).set_stroke(width=0)
            body.move_to(start)

            def _make_follow(_src, _xi, _yi):
                def updater(mob):
                    frame = current_frame()
                    mob.move_to(c2p(_src[frame, _xi], _src[frame, _yi]))

                return updater

            body.add_updater(_make_follow(src, x_idx, y_idx))
            self.add(body)

            if spec["zone_radius"] is not None:
                zone_color = _to_hex(spec["zone_color"])
                zone = Circle(radius=spec["zone_radius"] * unit_x, color=zone_color)
                zone.set_fill(zone_color, opacity=spec["zone_alpha"]).set_stroke(width=0)
                zone.move_to(start)
                zone.add_updater(_make_follow(src, x_idx, y_idx))
                self.add(zone)

            if spec["trail"]:
                trail = _build_reveal_group(
                    src[:, x_idx],
                    src[:, y_idx],
                    _to_hex(spec["trail_color"]),
                    style="line",
                    stroke_width=3,
                    opacity=spec["trail_alpha"],
                    dashed=spec["trail_style"] == "--",
                )
                self.add(trail)

        # --- per-frame predictions ---------------------------------------------
        for spec in self.predictions:
            self.add(self._prediction_group(spec, c2p, current_frame))

        # --- time overlay --------------------------------------------------------
        if self.show_time_overlay:
            time_text = Text("t = 0.0 s", font_size=20, color="#000000").to_corner(UL, buff=0.3)

            def _time_updater(mob):
                mob.become(
                    Text(
                        f"t = {current_frame() * self.dt:.1f} s", font_size=20, color="#000000"
                    ).to_corner(UL, buff=0.3)
                )

            time_text.add_updater(_time_updater)
            self.add(time_text)

        # --- play: real-time playback, clamped to a sane render length -----------
        playback_seconds = min(30.0, max(2.0, n_frames * self.dt))
        self.play(
            progress.animate(run_time=playback_seconds, rate_func=rate_functions.linear).set_value(
                n_frames - 1
            )
        )
        self.wait(0.5)

    # -----------------------------------------------------------------------
    def _build_axes(self):
        """Create 2D axes fitting the frame; returns (axes, unit_x, unit_y).

        ``unit_x`` / ``unit_y`` convert data units to scene units so element
        radii can be drawn to scale.  With ``aspect="equal"`` both units match.
        """
        from manim import Axes  # local import keeps module importable sans manim

        x_span = float(self.x_lim[1] - self.x_lim[0])
        y_span = float(self.y_lim[1] - self.y_lim[0])
        max_w, max_h = 12.0, 6.0

        if self.aspect == "equal":
            unit = min(max_w / x_span, max_h / y_span)
            x_len, y_len = x_span * unit, y_span * unit
        else:
            x_len, y_len = max_w, max_h

        axes = Axes(
            x_range=[self.x_lim[0], self.x_lim[1], max(x_span / 8, 1e-6)],
            y_range=[self.y_lim[0], self.y_lim[1], max(y_span / 8, 1e-6)],
            x_length=x_len,
            y_length=y_len,
            tips=False,
            axis_config={
                "include_ticks": True,
                "tick_size": 0.04,
                "stroke_width": 1.5,
                "color": "#555555",
            },
        )
        return axes, x_len / x_span, y_len / y_span

    # -----------------------------------------------------------------------
    def _prediction_group(self, spec: dict, c2p, current_frame):
        """A VGroup redrawn every frame from the shared prediction computer."""
        color = _to_hex(spec["color"])
        group = VGroup()

        def updater(mob):
            frame = current_frame()
            px, py = self.compute_prediction(spec, frame)
            segs = VGroup()
            n = len(px) - 1
            for k in range(n):
                seg = Line(
                    c2p(px[k], py[k]),
                    c2p(px[k + 1], py[k + 1]),
                    stroke_width=2 * spec["linewidth"],
                    color=color,
                )
                alpha = spec["alpha"] * (1.0 - k / n) if spec["fade"] else spec["alpha"]
                seg.set_opacity(alpha)
                segs.add(seg)
            mob.become(segs) if n >= 1 else mob.become(VGroup())

        group.add_updater(updater)
        return group

    # -----------------------------------------------------------------------
    def _synthetic_demo(self):
        """Simple goal-reaching demo so the scene renders standalone."""
        n = 100
        t = np.linspace(0, 1, n)
        states = np.stack([-3 + 6 * t, 1.5 * np.sin(2 * np.pi * t) * (1 - t)], axis=1)
        type(self).states = states
        type(self).goals = [{"position": (3.0, 0.0), "radius": 0.3, "color": "g", "label": "Goal"}]
        type(self).obstacles = [
            {"center": (0.0, 0.5), "radius": 0.5, "ellipse_radii": None, "color": "k", "alpha": 0.3}
        ]
        type(self).agents = [
            {
                "x_idx": 0,
                "y_idx": 1,
                "data": None,
                "body_radius": 0.2,
                "body_color": "blue",
                "body_alpha": 0.8,
                "zone_radius": None,
                "zone_color": "blue",
                "zone_alpha": 0.15,
                "trail": True,
                "trail_color": "blue",
                "trail_alpha": 0.5,
                "trail_style": "-",
                "label": "Agent",
                "zorder": 5,
            }
        ]
        type(self).show_time_overlay = True


# ---------------------------------------------------------------------------
# Mixin for CBFAnimator
# ---------------------------------------------------------------------------
class _ManimMixin:
    """Manim-specific build / save / show methods.

    Mixed into :class:`~cbfkit.utils.animators.animator.CBFAnimator`.
    Expects the host class to provide the element descriptor lists, limits,
    ``_dt``, ``_title``, ``_aspect``, ``_manim_quality``, and
    ``_compute_prediction``.
    """

    def _build_manim(self):
        """Inject animator state into :class:`CBFAnimator2DScene`; return it."""
        _require_manim()
        CBFAnimator2DScene.states = np.asarray(self._states)
        CBFAnimator2DScene.dt = self._dt
        CBFAnimator2DScene.x_lim = self._x_lim
        CBFAnimator2DScene.y_lim = self._y_lim
        CBFAnimator2DScene.title = self._title
        CBFAnimator2DScene.aspect = self._aspect
        CBFAnimator2DScene.goals = self._goals
        CBFAnimator2DScene.obstacles = self._obstacles
        CBFAnimator2DScene.trajectories = self._trajectories
        CBFAnimator2DScene.agents = self._agents
        CBFAnimator2DScene.predictions = self._predictions
        CBFAnimator2DScene.show_time_overlay = self._show_time
        CBFAnimator2DScene.compute_prediction = self._compute_prediction
        return CBFAnimator2DScene

    def _save_manim(self, path: str) -> str:
        """Render to *path* (``.mp4``, or ``.gif`` via Manim's gif format)."""
        import tempfile

        scene_cls = self._build_manim()

        overrides = {
            "quality": self._manim_quality,
            # Caching keys partial movie files by animation hash; with a fresh
            # media dir per render, stale in-process cache entries would point
            # at deleted files and crash a second render, so disable it.
            "disable_caching": True,
        }
        if path.lower().endswith(".gif"):
            overrides["format"] = "gif"
        # Render intermediates in a temp dir so the caller's cwd stays clean;
        # the finished video is copied to *path* below.  tempconfig scopes the
        # global Manim config so repeated renders in one process stay isolated.
        with tempfile.TemporaryDirectory(prefix="cbfkit_manim_") as tmp_media:
            overrides["media_dir"] = tmp_media
            with tempconfig(overrides):
                scene = scene_cls()
                scene.render()
            # For MP4 the file writer reports the exact output path; for GIF
            # it still reports the .mp4 name, so fall back to globbing the
            # media dir for the rendered file with the requested extension.
            rendered = str(scene.renderer.file_writer.movie_file_path)
            if not os.path.exists(rendered):
                import glob

                ext = os.path.splitext(path)[1] or ".mp4"
                candidates = glob.glob(
                    os.path.join(tmp_media, "videos", "**", f"*{ext}"), recursive=True
                )
                if not candidates:
                    raise FileNotFoundError(
                        f"Manim did not produce a {ext} file under {tmp_media!r}."
                    )
                rendered = candidates[0]

            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(rendered, path)
        return os.path.abspath(path)

    def _show_manim(self):
        """Render and open the result in the default player (Manim preview)."""
        scene_cls = self._build_manim()
        with tempconfig({"quality": self._manim_quality, "preview": True, "disable_caching": True}):
            scene = scene_cls()
            scene.render()
