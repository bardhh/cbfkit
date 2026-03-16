"""
Visualization Utilities for CBFKit Simulations.

* 2D crowd/trajectory animations delegate to
  :class:`cbfkit.utils.animator.CBFAnimator`.
* 3D multi-robot animations support Plotly (interactive HTML, default),
  matplotlib (MP4/GIF), and Manim (high-quality MP4) backends.
"""

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

_PED_COLORS = ["red", "orange", "purple", "brown", "pink"]

# Plotly tab10-equivalent colours
_PLOTLY_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


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


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "Optional dependency 'matplotlib' not found. "
            "Please install cbfkit[vis] to use visualization features."
        )


def _plot_ellipse_3d(ax, center, radii, rotation, color="blue", alpha=0.2):
    """Plot a 3D ellipsoid on *ax* (matplotlib)."""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)  # (3, n*n)
    pts = (rotation @ pts).T + center  # (n*n, 3)
    n = len(u)
    x = pts[:, 0].reshape(n, n)
    y = pts[:, 1].reshape(n, n)
    z = pts[:, 2].reshape(n, n)

    ax.plot_surface(x, y, z, color=color, rstride=4, cstride=4, alpha=alpha)


def _point_to_ellipsoid_distance(p, c, r, R):
    """Distance from point *p* to an ellipsoid (negative if inside)."""
    u = p - c
    norm_u = np.linalg.norm(u)
    if norm_u == 0:
        return -np.min(r)

    u_normalized = u / norm_u
    inv_r_squared = np.diag(1.0 / (r ** 2))
    K = u_normalized.T @ R @ inv_r_squared @ R.T @ u_normalized
    s = 1.0 / np.sqrt(K)
    x = c + s * u_normalized
    d = np.linalg.norm(p - x)
    return d if s >= 1 else -d


def _ellipsoid_mesh(center, radii, rotation, n=12):
    """Generate triangulated mesh vertices for a 3D ellipsoid (for Plotly)."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)  # (3, n*n)
    pts = (rotation @ pts).T + center  # (n*n, 3)

    # Build triangle indices for the grid (vectorized)
    rows, cols = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    p0 = (rows * n + cols).ravel()
    p1 = p0 + 1
    p2 = p0 + n
    p3 = p2 + 1
    ii = np.concatenate([p0, p0]).tolist()
    jj = np.concatenate([p1, p2]).tolist()
    kk = np.concatenate([p2, p3]).tolist()
    return pts[:, 0], pts[:, 1], pts[:, 2], ii, jj, kk


def _compute_distance_metrics(states, desired_states, num_robots, sdim,
                              ellipse_centers, ellipse_radii, ellipse_rotations,
                              include_min_dist, include_obs_dist):
    """Pre-compute distance arrays used by both backends."""
    N = len(states)
    time = np.arange(N) * 1.0  # placeholder, caller scales by dt

    # Goal distances (N, num_robots)
    goal_dists = np.zeros((N, num_robots))
    for i in range(num_robots):
        idx = sdim * i
        goal_dists[:, i] = np.linalg.norm(
            states[:, idx:idx + 3] - desired_states[idx:idx + 3], axis=1,
        )

    # Min inter-robot distances (vectorized over robots)
    min_dists = None
    if include_min_dist:
        min_dists = np.zeros((N, num_robots))
        for t in range(N):
            positions = states[t, :].reshape(num_robots, sdim)
            diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dists[t, :] = np.min(dists, axis=1)

    # Min obstacle distances
    obs_dists = None
    if include_obs_dist and ellipse_centers is not None:
        num_obs = len(ellipse_centers)
        obs_dists = np.zeros((N, num_robots))
        for t in range(N):
            for i in range(num_robots):
                idx = sdim * i
                p = states[t, idx:idx + 3]
                dmin = np.inf
                for j in range(num_obs):
                    d = _point_to_ellipsoid_distance(
                        p, ellipse_centers[j], ellipse_radii[j], ellipse_rotations[j],
                    )
                    dmin = min(dmin, d)
                obs_dists[t, i] = dmin

    return goal_dists, min_dists, obs_dists


# ---- Plotly 3D backend --------------------------------------------------

def _visualize_3d_plotly(
    states, desired_states, desired_state_radius, num_robots,
    ellipse_centers, ellipse_radii, ellipse_rotations,
    x_lim, y_lim, z_lim, dt, sdim, title, save_animation,
    animation_filename, include_min_distance_plot,
    include_min_distance_to_obstacles_plot, threshold,
    goal_dists, min_dists, obs_dists,
):
    from cbfkit.utils.animator import (
        _compute_plotly_frame_step,
        _plotly_animation_controls,
        _require_plotly,
    )
    _require_plotly()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    N = len(states)
    time = np.arange(N) * dt
    colors = [_PLOTLY_TAB10[i % len(_PLOTLY_TAB10)] for i in range(num_robots)]

    # 3D scenes are heavier to render; use a higher floor than 2D
    frame_indices, frame_duration_ms = _compute_plotly_frame_step(
        dt, N, max_frames=200, min_frame_ms=80,
    )

    # --- subplot layout ---
    num_cols = 2
    specs_row = [{"type": "scene"}, {"type": "xy"}]
    titles = [title, "Distance to Goal"]
    if include_min_distance_plot:
        num_cols += 1
        specs_row.append({"type": "xy"})
        titles.append("Min Dist Between Robots")
    if include_min_distance_to_obstacles_plot:
        num_cols += 1
        specs_row.append({"type": "xy"})
        titles.append("Min Dist to Obstacles")

    widths = [2] + [1] * (num_cols - 1)
    fig = make_subplots(
        rows=1, cols=num_cols,
        specs=[specs_row],
        column_widths=widths,
        subplot_titles=titles,
    )

    # --- static 3D elements ---
    # Goal markers + translucent sphere meshes
    for i in range(num_robots):
        idx = sdim * i
        gx, gy, gz = float(desired_states[idx]), float(desired_states[idx + 1]), float(desired_states[idx + 2])
        # Goal marker (large, prominent)
        fig.add_trace(
            go.Scatter3d(
                x=[gx], y=[gy], z=[gz],
                mode="markers",
                marker=dict(size=8, color=colors[i], symbol="diamond"),
                name=f"Goal {i + 1}",
                showlegend=True,
            ),
            row=1, col=1,
        )
        # Goal sphere (Mesh3d for proper rendering)
        sx, sy, sz, si, sj, sk = _ellipsoid_mesh(
            np.array([gx, gy, gz]),
            np.array([desired_state_radius] * 3),
            np.eye(3), n=15,
        )
        fig.add_trace(
            go.Mesh3d(
                x=sx.tolist(), y=sy.tolist(), z=sz.tolist(),
                i=si, j=sj, k=sk,
                color=colors[i], opacity=0.15,
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Ellipsoid obstacles
    if ellipse_centers is not None and ellipse_radii is not None and ellipse_rotations is not None:
        for ec, er, erot in zip(ellipse_centers, ellipse_radii, ellipse_rotations):
            mx, my, mz, ii, jj, kk = _ellipsoid_mesh(
                np.asarray(ec), np.asarray(er), np.asarray(erot),
            )
            fig.add_trace(
                go.Mesh3d(
                    x=mx.tolist(), y=my.tolist(), z=mz.tolist(),
                    i=ii, j=jj, k=kk,
                    color="black", opacity=0.2,
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # Count static traces so we know where animated traces start
    n_static = len(fig.data)

    # --- animated traces (initial = empty) ---
    # For each robot: 1 trajectory trace (3D) + 1 goal-dist trace (2D)
    # + optional min-dist + optional obs-dist
    for i in range(num_robots):
        # 3D trajectory
        fig.add_trace(
            go.Scatter3d(
                x=[], y=[], z=[],
                mode="lines",
                line=dict(color=colors[i], width=3),
                name=f"Robot {i + 1}",
                showlegend=True,
            ),
            row=1, col=1,
        )
        # Goal distance
        fig.add_trace(
            go.Scatter(
                x=[], y=[],
                mode="lines",
                line=dict(color=colors[i], width=2),
                name=f"Robot {i + 1}",
                showlegend=False,
            ),
            row=1, col=2,
        )

    if include_min_distance_plot:
        col_md = 3
        for i in range(num_robots):
            fig.add_trace(
                go.Scatter(
                    x=[], y=[],
                    mode="lines",
                    line=dict(color=colors[i], width=2),
                    name=f"Robot {i + 1}",
                    showlegend=False,
                ),
                row=1, col=col_md,
            )
        if threshold is not None:
            fig.add_hline(
                y=threshold, line_dash="dash", line_color="red",
                row=1, col=col_md,
            )

    if include_min_distance_to_obstacles_plot:
        col_od = 3 if not include_min_distance_plot else 4
        for i in range(num_robots):
            fig.add_trace(
                go.Scatter(
                    x=[], y=[],
                    mode="lines",
                    line=dict(color=colors[i], width=2),
                    name=f"Robot {i + 1}",
                    showlegend=False,
                ),
                row=1, col=col_od,
            )

    n_animated = len(fig.data) - n_static
    animated_indices = list(range(n_static, n_static + n_animated))

    # --- animation frames ---
    frames = []
    for fi in frame_indices:
        t = fi * dt
        trace_data = []

        for i in range(num_robots):
            idx = sdim * i
            # 3D trajectory up to frame fi
            trace_data.append(
                go.Scatter3d(
                    x=states[:fi, idx].tolist(),
                    y=states[:fi, idx + 1].tolist(),
                    z=states[:fi, idx + 2].tolist(),
                )
            )
            # Goal distance
            trace_data.append(
                go.Scatter(
                    x=time[:fi].tolist(),
                    y=goal_dists[:fi, i].tolist(),
                )
            )

        if include_min_distance_plot:
            for i in range(num_robots):
                trace_data.append(
                    go.Scatter(
                        x=time[:fi].tolist(),
                        y=min_dists[:fi, i].tolist(),
                    )
                )

        if include_min_distance_to_obstacles_plot:
            for i in range(num_robots):
                trace_data.append(
                    go.Scatter(
                        x=time[:fi].tolist(),
                        y=obs_dists[:fi, i].tolist(),
                    )
                )

        frames.append(
            go.Frame(
                data=trace_data,
                traces=animated_indices,
                name=f"{t:.2f}s",
            )
        )

    fig.frames = frames

    # --- scene + axis layout ---
    menus, sliders = _plotly_animation_controls(
        frames, frame_duration_ms, button_y=-0.25,
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=list(x_lim), title="X [m]"),
            yaxis=dict(range=list(y_lim), title="Y [m]"),
            zaxis=dict(range=list(z_lim), title="Z [m]"),
            aspectmode="cube",
        ),
        width=350 * num_cols,
        height=500,
        margin=dict(l=30, r=30, t=60, b=100),
        template="plotly_white",
        updatemenus=menus,
        sliders=sliders,
    )

    # 2D subplot axis labels
    fig.update_xaxes(title_text="Time [s]", row=1, col=2)
    fig.update_yaxes(title_text="Distance [m]", row=1, col=2)
    fig.update_yaxes(range=[0, float(np.max(goal_dists)) * 1.1], row=1, col=2)

    if include_min_distance_plot:
        fig.update_xaxes(title_text="Time [s]", row=1, col=3)
        fig.update_yaxes(title_text="Distance [m]", row=1, col=3)
        fig.update_yaxes(range=[0, float(np.max(min_dists)) * 1.1], row=1, col=3)

    if include_min_distance_to_obstacles_plot:
        col_od = 3 if not include_min_distance_plot else 4
        fig.update_xaxes(title_text="Time [s]", row=1, col=col_od)
        fig.update_yaxes(title_text="Distance [m]", row=1, col=col_od)
        fig.update_yaxes(range=[0, float(np.max(obs_dists)) * 1.1], row=1, col=col_od)

    # --- save / show ---
    if save_animation:
        output_path = Path(animation_filename).with_suffix(".html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), auto_open=False)
        abs_path = str(output_path.resolve())
        print(f"\nInteractive animation saved to: file://{abs_path}")
    else:
        fig.show()

    return fig


# ---- Matplotlib 3D backend -----------------------------------------------

def _visualize_3d_matplotlib(
    states, desired_states, desired_state_radius, num_robots,
    ellipse_centers, ellipse_radii, ellipse_rotations,
    x_lim, y_lim, z_lim, dt, sdim, title, save_animation,
    animation_filename, include_min_distance_plot,
    include_min_distance_to_obstacles_plot, threshold,
    goal_dists, min_dists, obs_dists,
):
    _require_matplotlib()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec

    from cbfkit.utils.animator import save_animation as _save_animation

    N = len(states)
    time = np.arange(N) * dt

    num_subplots = 2
    if include_min_distance_plot:
        num_subplots += 1
    if include_min_distance_to_obstacles_plot:
        num_subplots += 1

    ratios = [2] + [1] * (num_subplots - 1)
    fig = plt.figure(figsize=(4 + 4 * num_subplots, 6))
    gs = GridSpec(1, num_subplots, width_ratios=ratios)

    ax_traj = fig.add_subplot(gs[0], projection="3d")
    ax_traj.set_xlim(x_lim)
    ax_traj.set_ylim(y_lim)
    ax_traj.set_zlim(z_lim)
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.set_zlabel("Z [m]")
    ax_traj.set_title(title)
    ax_traj.grid(True)

    colors = plt.cm.get_cmap("tab10", num_robots).colors
    lines_traj = []
    lines_goal_dist = []

    ax_goal_dist = fig.add_subplot(gs[1])
    ax_goal_dist.set_xlim(0, time[-1])
    ax_goal_dist.set_xlabel("Time [s]")
    ax_goal_dist.set_ylabel("Distance to Goal [m]")
    ax_goal_dist.set_title("Distance to Desired States")
    ax_goal_dist.grid(True)
    ax_goal_dist.set_ylim(0, float(np.max(goal_dists)) * 1.1)

    for i in range(num_robots):
        idx = sdim * i
        color = colors[i % num_robots]

        ax_traj.scatter(
            desired_states[idx], desired_states[idx + 1], desired_states[idx + 2],
            color=color, s=50, label=f"Desired State {i + 1}",
        )
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        xs = desired_state_radius * np.outer(np.cos(u), np.sin(v)) + desired_states[idx]
        ys = desired_state_radius * np.outer(np.sin(u), np.sin(v)) + desired_states[idx + 1]
        zs = desired_state_radius * np.outer(np.ones_like(u), np.cos(v)) + desired_states[idx + 2]
        ax_traj.plot_wireframe(xs, ys, zs, color=color, linewidth=0.5, alpha=0.3)

        (lt,) = ax_traj.plot([], [], [], lw=2, label=f"Trajectory {i + 1}", color=color)
        lines_traj.append(lt)

        (lg,) = ax_goal_dist.plot([], [], lw=2, label=f"Robot {i + 1}", color=color)
        lines_goal_dist.append(lg)

    if ellipse_centers is not None and ellipse_radii is not None and ellipse_rotations is not None:
        for ec, er, erot in zip(ellipse_centers, ellipse_radii, ellipse_rotations):
            _plot_ellipse_3d(ax_traj, ec, er, erot, color="black", alpha=0.2)

    lines_min_dist = []
    if include_min_distance_plot:
        ax_min_dist = fig.add_subplot(gs[2])
        ax_min_dist.set_xlim(0, time[-1])
        ax_min_dist.set_xlabel("Time [s]")
        ax_min_dist.set_ylabel("Minimum Distance [m]")
        ax_min_dist.set_title("Min Distance Between Robots")
        ax_min_dist.grid(True)
        ax_min_dist.set_ylim(0, float(np.max(min_dists)) * 1.1)
        if threshold is not None:
            ax_min_dist.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
        for i in range(num_robots):
            (lm,) = ax_min_dist.plot([], [], lw=2, label=f"Robot {i + 1}", color=colors[i])
            lines_min_dist.append(lm)

    lines_obstacle_dist = []
    if include_min_distance_to_obstacles_plot:
        sp_idx = 2 if not include_min_distance_plot else 3
        ax_obs_dist = fig.add_subplot(gs[sp_idx])
        ax_obs_dist.set_xlim(0, time[-1])
        ax_obs_dist.set_xlabel("Time [s]")
        ax_obs_dist.set_ylabel("Distance to Obstacles [m]")
        ax_obs_dist.set_title("Min Distance to Obstacles")
        ax_obs_dist.grid(True)
        ax_obs_dist.set_ylim(0, float(np.max(obs_dists)) * 1.1)
        for i in range(num_robots):
            (lo,) = ax_obs_dist.plot([], [], lw=2, label=f"Robot {i + 1}", color=colors[i])
            lines_obstacle_dist.append(lo)

    def init():
        artists = []
        for lt in lines_traj:
            lt.set_data([], [])
            lt.set_3d_properties([])
            artists.append(lt)
        for lg in lines_goal_dist:
            lg.set_data([], [])
            artists.append(lg)
        for lm in lines_min_dist:
            lm.set_data([], [])
            artists.append(lm)
        for lo in lines_obstacle_dist:
            lo.set_data([], [])
            artists.append(lo)
        return artists

    def update(num):
        artists = []
        for i, (lt, lg) in enumerate(zip(lines_traj, lines_goal_dist)):
            idx = sdim * i
            lt.set_data(states[:num, idx], states[:num, idx + 1])
            lt.set_3d_properties(states[:num, idx + 2])
            artists.append(lt)
            lg.set_data(time[:num], goal_dists[:num, i])
            artists.append(lg)
        for i, lm in enumerate(lines_min_dist):
            lm.set_data(time[:num], min_dists[:num, i])
            artists.append(lm)
        for i, lo in enumerate(lines_obstacle_dist):
            lo.set_data(time[:num], obs_dists[:num, i])
            artists.append(lo)
        return artists

    ani = FuncAnimation(
        fig, update, frames=N,
        init_func=init, blit=True, interval=dt * 1000,
    )

    plt.tight_layout()

    if save_animation:
        Path(animation_filename).parent.mkdir(parents=True, exist_ok=True)
        _save_animation(ani, animation_filename)

    plt.show()

    axes = [ax_traj, ax_goal_dist]
    if include_min_distance_plot:
        axes.append(ax_min_dist)
    if include_min_distance_to_obstacles_plot:
        axes.append(ax_obs_dist)
    return fig, tuple(axes)


# ---- Manim 3D backend ----------------------------------------------------

def _visualize_3d_manim(
    states, desired_states, desired_state_radius, num_robots,
    ellipse_centers, ellipse_radii, ellipse_rotations,
    x_lim, y_lim, z_lim, dt, sdim, title, save_animation,
    animation_filename, include_min_distance_plot,
    include_min_distance_to_obstacles_plot, threshold,
    goal_dists, min_dists, obs_dists,
):
    import warnings

    from cbfkit.utils.animator import _require_manim
    from cbfkit.utils.visualizations.manim_3d_multi_robot import render_multi_robot_3d

    _require_manim()

    if include_min_distance_plot:
        warnings.warn(
            "include_min_distance_plot is not supported by the Manim backend and will be ignored.",
            stacklevel=3,
        )
    if include_min_distance_to_obstacles_plot:
        warnings.warn(
            "include_min_distance_to_obstacles_plot is not supported by the Manim backend "
            "and will be ignored.",
            stacklevel=3,
        )

    save_path = animation_filename if save_animation else None

    return render_multi_robot_3d(
        states=states,
        desired_states=desired_states,
        num_robots=num_robots,
        state_dimension_per_robot=sdim,
        desired_state_radius=desired_state_radius,
        x_lim=x_lim,
        y_lim=y_lim,
        z_lim=z_lim,
        dt=dt,
        title=title,
        ellipse_centers=ellipse_centers,
        ellipse_radii=ellipse_radii,
        ellipse_rotations=ellipse_rotations,
        save_path=save_path,
    )


# ---- Public entry point ---------------------------------------------------

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
    """
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
    # Use identical range on all axes so spheres aren't warped
    lo = min(float(xs.min()), float(ys.min()), float(zs.min())) - margin
    hi = max(float(xs.max()), float(ys.max()), float(zs.max())) + margin
    x_lim = y_lim = z_lim = (lo, hi)

    # Pre-compute metrics shared by both backends
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
        return _visualize_3d_plotly(save_animation=save_animation, **common)
    elif backend == "manim":
        return _visualize_3d_manim(save_animation=save_animation, **common)
    else:
        return _visualize_3d_matplotlib(save_animation=save_animation, **common)
