"""
Visualization Utilities for CBFKit Simulations.

* 2D crowd/trajectory animations delegate to
  :class:`cbfkit.utils.animator.CBFAnimator`.
* 3D multi-robot animations support Plotly (interactive HTML, default),
  matplotlib (MP4/GIF), and Manim (high-quality MP4) backends.
"""

from typing import Any, List, Optional

import numpy as np

_PED_COLORS = ["red", "orange", "purple", "brown", "pink"]

# Manim quality mapping
_MANIM_QUALITY_MAP = {
    "low": "low_quality",
    "medium": "medium_quality",
    "high": "high_quality",
    "production": "production_quality",
}


def visualize_crowd(
    states: np.ndarray,
    num_pedestrians: int,
    robot_goal: np.ndarray,
    d_safe: float = 1.0,
    dt: float = 0.1,
    p_values: Optional[List[Any]] = None,
    p_keys: Optional[List[str]] = None,
    save_path: str = "crowd_animation.mp4",
    backend: str = "plotly",
):
    """Generate an animation of the robot navigating among pedestrians.

    Parameters
    ----------
    states : np.ndarray
        Trajectory data with shape ``(N, dim)``.  The first 4 columns are
        the robot state ``[x, y, ...]``; subsequent groups of 4 columns are
        pedestrian states ``[px, py, vx, vy]``.
    num_pedestrians : int
        Number of pedestrians in *states*.
    robot_goal : np.ndarray
        Goal position ``[x, y, ...]``.
    d_safe : float
        Safety radius drawn around each pedestrian.
    dt : float
        Simulation time step.
    p_values, p_keys : list, optional
        Planner data for MPPI trajectory overlay.
    save_path : str
        Output filename.
    backend : str
        ``"plotly"`` (default) or ``"matplotlib"``.
    """
    from cbfkit.utils.animator import CBFAnimator

    print(f"Generating Animation -> {save_path}...")

    anim = CBFAnimator(
        states, dt=dt,
        title="Crowd Navigation",
        aspect="equal",
        backend=backend,
    )

    # Goal
    anim.add_goal(robot_goal[:2], radius=0.4, color="green", label="Goal")

    # Robot
    anim.add_agent(
        x_idx=0, y_idx=1,
        body_radius=0.3, body_color="blue", body_alpha=0.8,
        trail=True, trail_style="-", trail_alpha=0.5,
        label="Robot",
    )

    # Pedestrians
    for i in range(num_pedestrians):
        idx = 4 + i * 4
        color = _PED_COLORS[i % len(_PED_COLORS)]
        anim.add_agent(
            x_idx=idx, y_idx=idx + 1,
            body_radius=0.25, body_color=color, body_alpha=0.8,
            zone_radius=d_safe, zone_alpha=0.15,
            trail=True, trail_style="--", trail_alpha=0.3,
            label=f"Ped {i}",
        )
        anim.add_prediction(
            source="linear",
            agent_x_idx=idx, agent_y_idx=idx + 1,
            agent_vx_idx=idx + 2, agent_vy_idx=idx + 3,
            horizon=20,
            color=color, linestyle="dotted",
        )

    # MPPI planned path overlay
    if p_keys and "x_traj" in p_keys:
        mppi_idx = p_keys.index("x_traj")
        anim.add_prediction(
            source="data",
            trajectory_data=p_values[mppi_idx],
            traj_x_row=0, traj_y_row=1,
            color="green", linewidth=2.5, fade=True,
            label="Planned Path",
        )

    anim.show_time()
    anim.auto_limits(margin=1.5)
    anim.save(save_path)


# =========================================================================
# 3D Multi-Robot Visualization
# =========================================================================


def _parse_manim_backend(backend: str) -> str:
    """Parse ``"manim"`` or ``"manim-<quality>"`` and return a Manim quality string.

    Valid forms: ``"manim"``, ``"manim-low"``, ``"manim-medium"``,
    ``"manim-high"``, ``"manim-production"``.
    """
    if backend == "manim":
        return "low_quality"
    parts = backend.split("-", 1)
    if len(parts) == 2 and parts[0] == "manim" and parts[1] in _MANIM_QUALITY_MAP:
        return _MANIM_QUALITY_MAP[parts[1]]
    valid = ", ".join(f'"manim-{k}"' for k in _MANIM_QUALITY_MAP)
    raise ValueError(
        f"Unknown Manim backend {backend!r}. Use \"manim\" or one of: {valid}."
    )


def visualize_3d_multi_robot(
    states: np.ndarray,
    desired_states: np.ndarray,
    desired_state_radius: float,
    num_robots: int,
    ellipse_centers: Optional[List] = None,
    ellipse_radii: Optional[List] = None,
    ellipse_rotations: Optional[List] = None,
    x_lim=(-5, 5),
    y_lim=(-5, 5),
    z_lim=(-5, 5),
    dt: float = 0.1,
    state_dimension_per_robot: int = 3,
    title: str = "Multi-Robot Trajectory",
    save_animation: bool = False,
    animation_filename: str = "system_behavior.mp4",
    include_min_distance_plot: bool = False,
    include_min_distance_to_obstacles_plot: bool = False,
    threshold: Optional[float] = None,
    backend: str = "plotly",
):
    """Animate a 3D multi-robot system with optional distance subplots.

    Parameters
    ----------
    states : np.ndarray
        State trajectory ``(N, state_dim)``.
    desired_states : array-like
        Flat goal vector ``[x0, y0, z0, x1, y1, z1, ...]``.
    desired_state_radius : float
        Radius of the goal wireframe sphere.
    num_robots : int
        Number of robots.
    state_dimension_per_robot : int
        Columns per robot in *states* (default 3).
    backend : str
        ``"plotly"`` (default), ``"matplotlib"``, or ``"manim"``.
        Manim accepts a quality suffix: ``"manim-low"`` (default),
        ``"manim-medium"``, ``"manim-high"``, ``"manim-production"``.
    """
    from cbfkit.utils.visualizations.helpers_3d import _compute_distance_metrics

    states = np.asarray(states)
    desired_states = np.asarray(desired_states)
    sdim = state_dimension_per_robot

    if include_min_distance_to_obstacles_plot:
        if ellipse_centers is None or ellipse_radii is None or ellipse_rotations is None:
            raise ValueError(
                "ellipse_centers, ellipse_radii, and ellipse_rotations required "
                "for obstacle distance plot."
            )

    print(f"Generating 3D Animation -> {animation_filename}...")

    # Auto-compute axis limits from states + goals + obstacles
    margin = 2.0
    all_x, all_y, all_z = [], [], []
    for i in range(num_robots):
        idx = sdim * i
        all_x.append(states[:, idx])
        all_y.append(states[:, idx + 1])
        all_z.append(states[:, idx + 2])
        all_x.append(np.array([float(desired_states[idx])]))
        all_y.append(np.array([float(desired_states[idx + 1])]))
        all_z.append(np.array([float(desired_states[idx + 2])]))
    if ellipse_centers is not None:
        for ec in ellipse_centers:
            ec = np.asarray(ec)
            all_x.append(np.array([float(ec[0])]))
            all_y.append(np.array([float(ec[1])]))
            all_z.append(np.array([float(ec[2])]))
    xs, ys, zs = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_z)
    lo = min(float(xs.min()), float(ys.min()), float(zs.min())) - margin
    hi = max(float(xs.max()), float(ys.max()), float(zs.max())) + margin
    x_lim = y_lim = z_lim = (lo, hi)

    # Pre-compute metrics shared by all backends
    goal_dists, min_dists, obs_dists = _compute_distance_metrics(
        states, desired_states, num_robots, sdim,
        ellipse_centers, ellipse_radii, ellipse_rotations,
        include_min_distance_plot, include_min_distance_to_obstacles_plot,
    )

    common = dict(
        states=states, desired_states=desired_states,
        desired_state_radius=desired_state_radius, num_robots=num_robots,
        ellipse_centers=ellipse_centers, ellipse_radii=ellipse_radii,
        ellipse_rotations=ellipse_rotations,
        x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
        dt=dt, sdim=sdim, title=title,
        animation_filename=animation_filename,
        include_min_distance_plot=include_min_distance_plot,
        include_min_distance_to_obstacles_plot=include_min_distance_to_obstacles_plot,
        threshold=threshold,
        goal_dists=goal_dists, min_dists=min_dists, obs_dists=obs_dists,
    )

    if backend == "plotly":
        from cbfkit.utils.visualizations.plotly_3d_multi_robot import _visualize_3d_plotly
        return _visualize_3d_plotly(save_animation=save_animation, **common)
    elif backend.startswith("manim"):
        quality = _parse_manim_backend(backend)
        from cbfkit.utils.animators.deps import _require_manim
        from cbfkit.utils.visualizations.manim_3d_multi_robot import render_multi_robot_3d
        _require_manim()
        save_path = animation_filename if save_animation else None
        return render_multi_robot_3d(
            states=states,
            desired_states=desired_states,
            num_robots=num_robots,
            state_dimension_per_robot=sdim,
            desired_state_radius=desired_state_radius,
            x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
            dt=dt, title=title,
            ellipse_centers=ellipse_centers,
            ellipse_radii=ellipse_radii,
            ellipse_rotations=ellipse_rotations,
            save_path=save_path,
            quality=quality,
            goal_dists=goal_dists,
            min_dists=min_dists if include_min_distance_plot else None,
            obs_dists=obs_dists if include_min_distance_to_obstacles_plot else None,
        )
    else:
        from cbfkit.utils.visualizations.matplotlib_3d_multi_robot import _visualize_3d_matplotlib
        return _visualize_3d_matplotlib(save_animation=save_animation, **common)
