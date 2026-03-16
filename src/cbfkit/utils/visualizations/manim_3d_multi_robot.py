"""
Manim backend for 3D multi-robot trajectory animation.

Provides :class:`MultiRobot3DScene` and :func:`render_multi_robot_3d` for
high-quality 3D animations with smooth camera motion, safety bubbles, and
ellipsoidal obstacles.

Usage (standalone test with synthetic data):
    manim -pql src/cbfkit/utils/visualizations/manim_3d_multi_robot.py MultiRobot3DScene

Usage (from library):
    from cbfkit.utils.visualization import visualize_3d_multi_robot
    visualize_3d_multi_robot(..., backend="manim")
"""

from __future__ import annotations

import numpy as np

from cbfkit.utils.animator import _require_manim

try:
    from manim import (
        PI,
        config,
        Dot3D,
        Sphere,
        ThreeDAxes,
        ThreeDScene,
        TracedPath,
        ValueTracker,
        rate_functions,
    )

    _MANIM_AVAILABLE = True
except ImportError:
    _MANIM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colour palette (matches matplotlib tab10 order)
# ---------------------------------------------------------------------------
_TAB10_HEX = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _robot_color(index: int) -> str:
    return _TAB10_HEX[index % len(_TAB10_HEX)]


# ---------------------------------------------------------------------------
# Helper: build a goal sphere (wireframe-style)
# ---------------------------------------------------------------------------
def _goal_sphere(center: np.ndarray, radius: float, color: str) -> "Sphere":
    return Sphere(center=center, radius=radius).set_opacity(0.25).set_color(color)


# ---------------------------------------------------------------------------
# Core Manim Scene
# ---------------------------------------------------------------------------
class MultiRobot3DScene(ThreeDScene):
    """Render multi-robot 3D trajectories.

    Pass data via class attributes before calling ``scene.render()``, or let
    the scene generate synthetic demo data when run standalone.
    """

    # --- class-level defaults (overridden by render_multi_robot_3d) ---------
    states: np.ndarray | None = None
    desired_states: np.ndarray | None = None
    num_robots: int = 2
    state_dim_per_robot: int = 6
    desired_state_radius: float = 0.3
    safety_radius: float = 0.25  # per-robot collision avoidance bubble
    x_lim: tuple[float, float] = (-15, 15)
    y_lim: tuple[float, float] = (-15, 15)
    z_lim: tuple[float, float] = (-10, 10)
    dt: float = 0.05
    title: str = "Multi-Robot Trajectory"
    # Static ellipsoidal obstacles: lists of centers, radii, rotation matrices
    ellipse_centers: list[np.ndarray] | None = None
    ellipse_radii: list[np.ndarray] | None = None
    ellipse_rotations: list[np.ndarray] | None = None
    # Scale factor to fit data into Manim's coordinate system (default ~7 units)
    _scale: float = 1.0

    # -----------------------------------------------------------------------
    def construct(self):
        # --- Synthetic demo data if none provided --------------------------
        if self.states is None:
            self.states, self.desired_states = self._synthetic_demo()

        states = np.asarray(self.states)
        goals = np.asarray(self.desired_states)
        n_robots = self.num_robots
        sdim = self.state_dim_per_robot
        n_frames = len(states)

        # Auto-scale: map the data range into roughly [-7, 7] Manim units
        all_pos = []
        for i in range(n_robots):
            idx = sdim * i
            all_pos.append(states[:, idx : idx + 3])
            all_pos.append(goals[idx : idx + 3].reshape(1, 3))
        all_pos = np.concatenate(all_pos, axis=0)
        data_extent = max(np.ptp(all_pos, axis=0))  # largest range
        self._scale = 7.0 / max(data_extent, 1e-6)

        def s(p: np.ndarray) -> np.ndarray:
            """Scale a 3-vector into Manim coordinates."""
            return np.array(p[:3], dtype=float) * self._scale

        # --- Axes ----------------------------------------------------------
        axis_len = 7.5
        axes = ThreeDAxes(
            x_range=[-axis_len, axis_len, 2],
            y_range=[-axis_len, axis_len, 2],
            z_range=[-axis_len, axis_len, 2],
            x_length=2 * axis_len,
            y_length=2 * axis_len,
            z_length=2 * axis_len,
        )

        self.set_camera_orientation(phi=70 * PI / 180, theta=-45 * PI / 180)
        self.add(axes)

        # --- Static ellipsoidal obstacles -----------------------------------
        if (
            self.ellipse_centers is not None
            and self.ellipse_radii is not None
            and self.ellipse_rotations is not None
        ):
            for center, radii, rotation in zip(
                self.ellipse_centers, self.ellipse_radii, self.ellipse_rotations
            ):
                sc = s(center)
                # Approximate ellipsoid as a Sphere scaled along each axis
                avg_r = float(np.mean(radii)) * self._scale
                obstacle = Sphere(center=sc, radius=avg_r)
                obstacle.set_color("#aa0000").set_opacity(0.45)
                # Stretch to match ellipsoid radii
                sr = np.array(radii, dtype=float) * self._scale
                if avg_r > 1e-8:
                    obstacle.stretch(sr[0] / avg_r, 0)
                    obstacle.stretch(sr[1] / avg_r, 1)
                    obstacle.stretch(sr[2] / avg_r, 2)
                # Apply rotation (Manim uses column-major 4x4)
                rot = np.eye(4)
                rot[:3, :3] = np.array(rotation)
                obstacle.apply_matrix(rot[:3, :3])
                obstacle.move_to(sc)
                self.add(obstacle)

        # --- Goal markers --------------------------------------------------
        for i in range(n_robots):
            idx = sdim * i
            goal_pos = s(goals[idx : idx + 3])
            color = _robot_color(i)
            goal_dot = Dot3D(point=goal_pos, radius=0.15, color=color).set_opacity(0.9)
            goal_sphere = _goal_sphere(
                goal_pos, self.desired_state_radius * self._scale, color
            )
            self.add(goal_dot, goal_sphere)

        # --- Animated robot dots + safety bubbles + traced paths -----------
        progress = ValueTracker(0)  # 0 -> n_frames-1
        safety_r_scaled = self.safety_radius * self._scale

        robot_dots = []
        safety_bubbles = []
        traced_paths = []

        for i in range(n_robots):
            idx = sdim * i
            color = _robot_color(i)
            start = s(states[0, idx : idx + 3])

            dot = Dot3D(point=start, radius=0.12, color=color)

            # Safety bubble: translucent sphere showing collision avoidance radius
            bubble = Sphere(center=start, radius=safety_r_scaled)
            bubble.set_color(color).set_opacity(0.12)

            # Updater: move dot and bubble to the frame indicated by progress
            def _make_updater(_i, _idx):
                def updater(mob):
                    frame = int(progress.get_value())
                    frame = min(frame, n_frames - 1)
                    pos = s(states[frame, _idx : _idx + 3])
                    mob.move_to(pos)

                return updater

            dot.add_updater(_make_updater(i, idx))
            bubble.add_updater(_make_updater(i, idx))

            # TracedPath draws the trail behind the dot
            path = TracedPath(
                dot.get_center,
                stroke_color=color,
                stroke_width=3,
                dissipating_time=None,  # keep full trail
            )

            robot_dots.append(dot)
            safety_bubbles.append(bubble)
            traced_paths.append(path)
            self.add(dot, bubble, path)

        # --- Camera rotation during playback --------------------------------
        self.begin_ambient_camera_rotation(rate=0.08)

        # --- Animate progress tracker from 0 -> n_frames-1 ------------------
        # Use ~10 s of playback regardless of sim length
        playback_seconds = min(15, max(5, n_frames * self.dt))
        self.play(
            progress.animate(run_time=playback_seconds, rate_func=rate_functions.linear).set_value(
                n_frames - 1
            ),
        )

        self.stop_ambient_camera_rotation()
        self.wait(1)

    # -----------------------------------------------------------------------
    def _synthetic_demo(self):
        """Generate simple circular swap trajectories for 2 robots + obstacle."""
        n_robots = 2
        sdim = 6
        n_steps = 300
        states = np.zeros((n_steps, sdim * n_robots))
        goals = np.zeros(sdim * n_robots)

        for i in range(n_robots):
            angle0 = 2 * np.pi * i / n_robots
            idx = sdim * i
            r = 10.0
            x0, y0 = r * np.cos(angle0), r * np.sin(angle0)
            goals[idx], goals[idx + 1], goals[idx + 2] = -x0, -y0, 0.0

            for t in range(n_steps):
                frac = t / (n_steps - 1)
                ang = angle0 + np.pi * frac
                states[t, idx] = r * np.cos(ang)
                states[t, idx + 1] = r * np.sin(ang)
                states[t, idx + 2] = 3 * np.sin(2 * np.pi * frac)

        # Add a demo obstacle at the origin
        self.ellipse_centers = [np.array([0.0, 0.0, 0.0])]
        self.ellipse_radii = [np.array([2.0, 2.0, 2.0])]
        self.ellipse_rotations = [np.eye(3)]
        self.safety_radius = 1.0

        return states, goals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_multi_robot_3d(
    states: np.ndarray,
    desired_states: np.ndarray,
    num_robots: int = 2,
    state_dimension_per_robot: int = 6,
    desired_state_radius: float = 0.3,
    safety_radius: float = 0.25,
    x_lim: tuple[float, float] = (-15, 15),
    y_lim: tuple[float, float] = (-15, 15),
    z_lim: tuple[float, float] = (-10, 10),
    dt: float = 0.05,
    title: str = "Multi-Robot Trajectory",
    ellipse_centers: list[np.ndarray] | None = None,
    ellipse_radii: list[np.ndarray] | None = None,
    ellipse_rotations: list[np.ndarray] | None = None,
    save_path: str | None = None,
    quality: str = "low_quality",
) -> str:
    """Render a multi-robot 3D animation using Manim.

    Parameters
    ----------
    states : np.ndarray
        ``(N, state_dim)`` array of simulation states.
    desired_states : np.ndarray
        ``(state_dim,)`` goal vector.
    safety_radius : float
        Per-robot collision avoidance bubble radius.
    ellipse_centers : list, optional
        List of obstacle center positions ``[x, y, z]``.
    ellipse_radii : list, optional
        List of obstacle semi-axis lengths ``[rx, ry, rz]``.
    ellipse_rotations : list, optional
        List of 3x3 rotation matrices for obstacles.
    save_path : str, optional
        If given, the output video is copied here.
    quality : str
        Manim quality flag: ``"low_quality"``, ``"medium_quality"``,
        ``"high_quality"``, or ``"production_quality"``.

    Returns
    -------
    str
        Path to the rendered video file.
    """
    _require_manim()

    import os

    # Configure Manim output
    config.quality = quality
    if save_path:
        config.output_file = os.path.basename(save_path)
        config.media_dir = os.path.dirname(save_path) or "./media"

    # Inject data into the scene class
    MultiRobot3DScene.states = np.asarray(states)
    MultiRobot3DScene.desired_states = np.asarray(desired_states)
    MultiRobot3DScene.num_robots = num_robots
    MultiRobot3DScene.state_dim_per_robot = state_dimension_per_robot
    MultiRobot3DScene.desired_state_radius = desired_state_radius
    MultiRobot3DScene.safety_radius = safety_radius
    MultiRobot3DScene.x_lim = x_lim
    MultiRobot3DScene.y_lim = y_lim
    MultiRobot3DScene.z_lim = z_lim
    MultiRobot3DScene.dt = dt
    MultiRobot3DScene.title = title
    MultiRobot3DScene.ellipse_centers = ellipse_centers
    MultiRobot3DScene.ellipse_radii = ellipse_radii
    MultiRobot3DScene.ellipse_rotations = ellipse_rotations

    scene = MultiRobot3DScene()
    scene.render()

    # Return path to rendered file
    return str(scene.renderer.file_writer.movie_file_path)
