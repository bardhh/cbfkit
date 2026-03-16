"""Matplotlib 3D multi-robot visualization backend."""

from pathlib import Path

import numpy as np

from .helpers_3d import _plot_ellipse_3d


def _visualize_3d_matplotlib(
    states, desired_states, desired_state_radius, num_robots,
    ellipse_centers, ellipse_radii, ellipse_rotations,
    x_lim, y_lim, z_lim, dt, sdim, title, save_animation,
    animation_filename, include_min_distance_plot,
    include_min_distance_to_obstacles_plot, threshold,
    goal_dists, min_dists, obs_dists,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec

    from cbfkit.utils.animators.helpers import save_animation as _save_animation

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
