"""
Visualization Utilities for Crowd and Robot Simulations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def require_visualization():
    """Raise an ImportError if matplotlib is not installed."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Optional dependency 'matplotlib' not found. "
            "Please install cbfkit[vis] to use visualization features."
        )


def get_fading_segments(x, y):
    """Helper to create segments and alphas for fading line."""
    if len(x) < 2:
        return [], []
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Alphas: 1.0 at start, 0.0 at end
    alphas = np.linspace(1.0, 0.0, len(segments))
    return segments, alphas


def visualize_crowd(
    states: np.ndarray,
    num_pedestrians: int,
    robot_goal: np.ndarray,
    d_safe: float = 1.0,
    dt: float = 0.1,
    p_values: Optional[List[Any]] = None,
    p_keys: Optional[List[str]] = None,
    save_path: str = "crowd_animation.mp4",
):
    """
    Generates an animation of the robot and pedestrians.

    Args:
        states (np.ndarray): Trajectory data [N, dim].
        num_pedestrians (int): Number of pedestrians.
        robot_goal (np.ndarray): Goal position [x, y].
        d_safe (float): Safety radius for visualization.
        dt (float): Time step.
        p_values (list, optional): Planner data values (for MPPI viz).
        p_keys (list, optional): Planner data keys.
        save_path (str): Output filename.
    """
    require_visualization()
    print(f"Generating Animation -> {save_path}...")
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Extract Robot State (Assume first 4 indices)
    rx = states[:, 0]
    ry = states[:, 1]

    # Extract Pedestrians
    # Assume indices start after robot (index 4) and are 4-dim each
    hx, hy, hvx, hvy = [], [], [], []
    for i in range(num_pedestrians):
        idx = 4 + i * 4
        hx.append(states[:, idx])
        hy.append(states[:, idx + 1])
        hvx.append(states[:, idx + 2])
        hvy.append(states[:, idx + 3])

    # Set Plot Limits
    all_x = np.concatenate([rx] + hx + [np.array([robot_goal[0]])])
    all_y = np.concatenate([ry] + hy + [np.array([robot_goal[1]])])
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # --- Static Elements ---
    # Goal
    ax.add_patch(
        Circle((robot_goal[0], robot_goal[1]), 0.4, color="green", alpha=0.5, label="Goal")
    )

    # --- Dynamic Elements (Patches) ---
    robot_patch = Circle((0, 0), 0.3, color="blue", label="Robot")
    ax.add_patch(robot_patch)
    (robot_trail,) = ax.plot([], [], "b-", alpha=0.5)

    ped_patches = []
    ped_trails = []
    ped_preds = []  # LineCollections for predictions
    colors = ["red", "orange", "purple", "brown", "pink"]

    for i in range(num_pedestrians):
        color = colors[i % len(colors)]
        # Safety Zone
        c_zone = Circle((0, 0), d_safe, color=color, alpha=0.15)
        # Body
        c_body = Circle((0, 0), 0.25, color=color, alpha=0.8)
        ax.add_patch(c_zone)
        ax.add_patch(c_body)
        ped_patches.append((c_zone, c_body))

        # Trail
        (tr,) = ax.plot([], [], "--", color=color, alpha=0.3)
        ped_trails.append(tr)

        # Prediction LineCollection
        lc = LineCollection([], linewidths=1.5, linestyles="dotted", colors=color)
        ax.add_collection(lc)
        ped_preds.append(lc)

    # MPPI Prediction
    mppi_lc = LineCollection([], linewidths=2.5, colors="green", alpha=0.8, label="Planned Path")
    ax.add_collection(mppi_lc)
    mppi_idx = -1
    if p_keys and "x_traj" in p_keys:
        mppi_idx = p_keys.index("x_traj")

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # --- Animation Function ---
    def animate(frame):
        t = frame * dt

        # Robot
        robot_patch.center = (rx[frame], ry[frame])
        robot_trail.set_data(rx[:frame], ry[:frame])

        # Pedestrians
        for i in range(num_pedestrians):
            zone, body = ped_patches[i]
            pos = (hx[i][frame], hy[i][frame])
            vel = (hvx[i][frame], hvy[i][frame])

            zone.center = pos
            body.center = pos
            ped_trails[i].set_data(hx[i][:frame], hy[i][:frame])

            # Simple Linear Prediction for Visualization
            # (Ideally we'd visualize the SFM prediction if we had access to it,
            # but linear is a good baseline for what "might" happen)
            pred_horizon = 20
            pred_x = [pos[0] + vel[0] * dt * k for k in range(pred_horizon)]
            pred_y = [pos[1] + vel[1] * dt * k for k in range(pred_horizon)]

            segs, alphas = get_fading_segments(pred_x, pred_y)
            ped_preds[i].set_segments(segs)
            ped_preds[i].set_alpha(alphas)

        # MPPI Plan
        if mppi_idx >= 0 and p_values and frame < len(p_values[mppi_idx]):
            traj = p_values[mppi_idx][frame]
            # Expecting shape (StateDim, Horizon) or similar.
            # Previous fix showed it was [0, :] and [1, :] for x/y.
            if traj.ndim >= 2 and traj.shape[0] >= 2:
                segs, alphas = get_fading_segments(traj[0, :], traj[1, :])
                mppi_lc.set_segments(segs)
                mppi_lc.set_alpha(alphas)
        else:
            mppi_lc.set_segments([])

        time_text.set_text(f"Time: {t:.1f}s")

        artists = [robot_patch, robot_trail, mppi_lc, time_text]
        for pair in ped_patches:
            artists.extend(pair)
        artists.extend(ped_trails)
        artists.extend(ped_preds)
        return artists

    anim = animation.FuncAnimation(fig, animate, frames=len(states), interval=50, blit=True)

    from cbfkit.utils.animator import save_animation

    save_animation(anim, str(output_path))

    plt.close()
