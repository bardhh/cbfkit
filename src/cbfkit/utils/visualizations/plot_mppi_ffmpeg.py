"""MPPI trajectory animation with circle obstacles."""

import matplotlib.pyplot as plt
import numpy as np


class circle:
    """Circle visualizer class."""

    def __init__(self, ax, pos=np.array([0, 0]), radius=1.0):
        """Initialize circle."""
        self.X = pos
        self.radius = radius
        self.id = id
        self.type = "circle"

        self.render(ax)

    def render(self, ax):
        """Render the circle on the axes."""
        circ = plt.Circle(
            (self.X[0], self.X[1]), self.radius, linewidth=1, edgecolor="k", facecolor="k"
        )
        ax.add_patch(circ)


#! PLOTTING
def plot_trajectory(
    states,
    title=None,
):
    """Plot the trajectory of the system states."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    lbl = ["X", "Y", "Z"]
    for ii in range(3):
        axs[ii].plot(states[:, ii], label=lbl[ii])

    plt.show()

    return fig, axs


#! ANIMATIONS
def animate(
    states,
    estimates,
    controller_data_keys,
    controller_data_items,
    mppi_args,
    desired_state,
    desired_state_radius,
    x_lim=(0, 10),
    y_lim=(0, 10),
    dt=0.1,
    title="System Behavior",
    save_animation=False,
    animation_filename="system_behavior.gif",
    obstacle=None,
    obstacle_radius=None,
    goal_radius=None,
    backend="matplotlib",
):
    """Animate the system behavior with MPPI trajectories.

    Parameters
    ----------
    backend : str
        ``"matplotlib"`` for MP4/GIF with MPPI sampled trajectories, or
        ``"plotly"`` for interactive HTML (trajectory only).
    """
    from cbfkit.utils.animator import AnimationConfig, CBFAnimator

    states_np = np.asarray(states)
    estimates_np = np.asarray(estimates)

    animator = CBFAnimator(
        states_np,
        dt=dt,
        x_lim=x_lim,
        y_lim=y_lim,
        title=title,
        backend=backend,
        config=AnimationConfig(blit=False) if backend == "matplotlib" else None,
    )

    # Goals (desired_state may be a list of waypoints)
    if hasattr(desired_state, "__len__") and np.ndim(desired_state) > 1:
        for k, ds in enumerate(desired_state):
            animator.add_goal(ds[:2], radius=desired_state_radius, label=f"Goal {k + 1}")
    else:
        animator.add_goal(desired_state[:2], radius=desired_state_radius)

    if obstacle is not None:
        animator.add_obstacle(obstacle, radius=obstacle_radius)

    animator.add_trajectory(x_idx=0, y_idx=1, color="k", label="Trajectory")
    animator.show_time()

    # MPPI overlay (matplotlib only)
    if backend == "matplotlib":
        fig, ax = animator.build()

        sampled_lines = []
        for _ in range(mppi_args["plot_samples"]):
            (line,) = ax.plot([], [], "g", alpha=0.2)
            sampled_lines.append(line)
        (selected_line,) = ax.plot([], [], "b", linewidth=2)

        sampled_key = (
            "sampled_x_traj"
            if "sampled_x_traj" in controller_data_keys
            else "robot_sampled_states"
        )
        state_dim = mppi_args["robot_state_dim"]

        def mppi_overlay(frame, _ax):
            # Note: this file uses [frame][key] indexing (not [key][frame])
            if frame >= len(controller_data_items):
                return []
            frame_data = controller_data_items[frame]
            sampled = frame_data[controller_data_keys.index(sampled_key)]
            selected = frame_data[controller_data_keys.index("x_traj")]

            for i, line in enumerate(sampled_lines):
                line.set_data(
                    sampled[state_dim * i, :],
                    sampled[state_dim * i + 1, :],
                )
            selected_line.set_data(selected[0, :], selected[1, :])

            return sampled_lines + [selected_line]

        animator.on_frame(mppi_overlay)

    if save_animation:
        animator.save(animation_filename)

    if backend == "matplotlib":
        animator.show()

    return animator.fig, animator.ax
