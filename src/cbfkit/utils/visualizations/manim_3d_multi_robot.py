"""
Manim backend for 3D multi-robot trajectory animation.

Provides :class:`MultiRobot3DScene` and :func:`render_multi_robot_3d` for
high-quality 3D animations with smooth camera motion, safety bubbles, and
ellipsoidal obstacles.  Side panels show distance-to-goal, min inter-robot
distance, and min obstacle distance per robot.

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
        DOWN,
        PI,
        RIGHT,
        UP,
        Axes,
        Dot3D,
        Line,
        Sphere,
        Text,
        ThreeDAxes,
        ThreeDScene,
        TracedPath,
        VGroup,
        ValueTracker,
        config,
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
# Helper: build a 2D chart panel as a fixed-screen overlay
# ---------------------------------------------------------------------------
def _build_chart_panel(
    title_text: str,
    data: np.ndarray,
    dt: float,
    num_robots: int,
    panel_width: float = 3.8,
    panel_height: float = 2.0,
):
    """Build a 2D Axes with pre-drawn static line segments for each robot.

    Returns ``(axes, title_mob, robot_lines)`` where ``robot_lines[i]`` is a
    :class:`VGroup` of ``Line`` segments for robot *i* that will be revealed
    progressively by an updater.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(N, num_robots)`` — one column per robot.
    """
    N = len(data)
    t_max = (N - 1) * dt
    y_max = float(np.nanmax(data)) * 1.15
    if y_max < 1e-6:
        y_max = 1.0

    axes = Axes(
        x_range=[0, t_max, t_max / 4],
        y_range=[0, y_max, y_max / 4],
        x_length=panel_width,
        y_length=panel_height,
        axis_config={"include_ticks": True, "tick_size": 0.05, "stroke_width": 1.5},
    )

    title_mob = Text(title_text, font_size=18).next_to(axes, UP, buff=0.12)

    # x-axis label
    x_label = Text("t [s]", font_size=14).next_to(axes, DOWN, buff=0.15)

    # Pre-build line segments per robot (initially invisible)
    robot_lines: list[VGroup] = []
    for r in range(num_robots):
        color = _robot_color(r)
        segs = VGroup()
        for k in range(N - 1):
            t0, t1 = k * dt, (k + 1) * dt
            y0, y1 = float(data[k, r]), float(data[k + 1, r])
            p0 = axes.c2p(t0, y0)
            p1 = axes.c2p(t1, y1)
            seg = Line(p0, p1, stroke_width=2, color=color)
            seg.set_opacity(0)
            segs.add(seg)
        robot_lines.append(segs)

    panel = VGroup(axes, title_mob, x_label, *robot_lines)
    return axes, title_mob, x_label, robot_lines, panel


# ---------------------------------------------------------------------------
# Core Manim Scene
# ---------------------------------------------------------------------------
class MultiRobot3DScene(ThreeDScene):
    """Render multi-robot 3D trajectories with distance metric side panels.

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
    # Static ellipsoidal obstacles: lists of centres, radii, rotation matrices
    ellipse_centers: list[np.ndarray] | None = None
    ellipse_radii: list[np.ndarray] | None = None
    ellipse_rotations: list[np.ndarray] | None = None
    # Pre-computed distance metrics (set by render_multi_robot_3d)
    goal_dists: np.ndarray | None = None
    min_dists: np.ndarray | None = None
    obs_dists: np.ndarray | None = None
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

        # --- Determine which side panels to show ---------------------------
        has_goal = self.goal_dists is not None
        has_min = self.min_dists is not None
        has_obs = self.obs_dists is not None
        n_panels = int(has_goal) + int(has_min) + int(has_obs)

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

        # --- Build side-panel charts (fixed to camera) ---------------------
        panels_group = VGroup()
        all_robot_lines: list[list[VGroup]] = []  # per-panel list of robot lines

        panel_configs = []
        if has_goal:
            panel_configs.append(("Dist to Goal", self.goal_dists))
        if has_min:
            panel_configs.append(("Min Inter-Robot Dist", self.min_dists))
        if has_obs:
            panel_configs.append(("Min Obstacle Dist", self.obs_dists))

        panel_height = min(2.0, 5.5 / max(n_panels, 1))
        panel_gap = 0.4

        for p_idx, (p_title, p_data) in enumerate(panel_configs):
            _, _, _, robot_lines, panel = _build_chart_panel(
                title_text=p_title,
                data=p_data,
                dt=self.dt,
                num_robots=n_robots,
                panel_width=3.8,
                panel_height=panel_height,
            )
            all_robot_lines.append(robot_lines)
            panels_group.add(panel)

        # Stack panels vertically and position to the right
        if n_panels > 0:
            panels_group.arrange(DOWN, buff=panel_gap)
            panels_group.to_edge(RIGHT, buff=0.1)
            # Move up slightly to centre vertically
            panels_group.shift(UP * 0.2)

            # Charts are 2D screen-space overlays — fix to camera
            self.add_fixed_in_frame_mobjects(panels_group)

            # Build updaters for progressive line reveal
            def _make_line_updater(robot_lines_for_panel, _n_frames):
                def _updater(mob):
                    frame = int(progress.get_value())
                    frame = min(frame, _n_frames - 1)
                    for rl in robot_lines_for_panel:
                        for k, seg in enumerate(rl):
                            seg.set_opacity(1.0 if k < frame else 0.0)

                return _updater

            for robot_lines in all_robot_lines:
                # Attach updater to the first robot's line group (triggers for all)
                dummy = robot_lines[0]
                dummy.add_updater(_make_line_updater(robot_lines, n_frames))

        # --- Camera rotation during playback --------------------------------
        self.begin_ambient_camera_rotation(rate=0.08)

        # --- Animate progress tracker from 0 -> n_frames-1 ------------------
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

        # Compute demo metrics
        self.goal_dists = np.zeros((n_steps, n_robots))
        for i in range(n_robots):
            idx = sdim * i
            self.goal_dists[:, i] = np.linalg.norm(
                states[:, idx : idx + 3] - goals[idx : idx + 3], axis=1
            )
        self.min_dists = np.full((n_steps, n_robots), 20.0)  # placeholder
        for t in range(n_steps):
            d = np.linalg.norm(states[t, :3] - states[t, sdim : sdim + 3])
            self.min_dists[t, :] = d

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
    goal_dists: np.ndarray | None = None,
    min_dists: np.ndarray | None = None,
    obs_dists: np.ndarray | None = None,
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
    goal_dists : np.ndarray, optional
        ``(N, num_robots)`` distance-to-goal per robot per step.
    min_dists : np.ndarray, optional
        ``(N, num_robots)`` min inter-robot distance per robot per step.
    obs_dists : np.ndarray, optional
        ``(N, num_robots)`` min obstacle distance per robot per step.

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
    MultiRobot3DScene.goal_dists = goal_dists
    MultiRobot3DScene.min_dists = min_dists
    MultiRobot3DScene.obs_dists = obs_dists

    scene = MultiRobot3DScene()
    scene.render()

    # Return path to rendered file
    return str(scene.renderer.file_writer.movie_file_path)
