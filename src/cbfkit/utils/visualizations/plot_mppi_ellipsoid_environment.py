"""MPPI trajectory animation in an ellipsoidal obstacle environment."""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


#! PLOTTING
def plot_trajectory(
    states,
    desired_state,
    desired_state_radius=0.25,
    obstacles=[],
    ellipsoids=[],
    x_lim=(-4, 4),
    y_lim=(-4, 4),
    title="System Behavior",
    fig=None,
    ax=None,
    savefile=None,
):
    """Plot the trajectory with ellipsoidal obstacles."""
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.plot(desired_state[0], desired_state[1], "ro", markersize=5, label="desired_state")
    ax.add_patch(
        plt.Circle(
            desired_state,
            desired_state_radius,
            color="r",
            fill=False,
            linestyle="--",
            linewidth=1,
        )
    )
    for obs, ell in zip(obstacles, ellipsoids):
        ax.add_patch(
            Ellipse(
                (obs[0], obs[1]),
                width=ell[0] * 2,
                height=ell[1] * 2,
                facecolor="k",
            )
        )

    ax.plot(states[:, 0], states[:, 1], label="Trajectory")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2])
    ax.grid()

    if savefile is not None:
        fig.savefig(savefile)

    return fig, ax


#! ANIMATIONS
def animate(
    states,
    estimates,
    controller_data_keys,
    controller_data_items,
    mppi_args,
    desired_state,
    desired_state_radius=0.1,
    obstacles=[],
    ellipsoids=[],
    x_lim=(-4, 4),
    y_lim=(-4, 4),
    dt=0.1,
    title="System Behavior",
    save_animation=True,
    animation_filename="system_behavior.gif",
    backend="matplotlib",
):
    """Animate the system behavior in an ellipsoidal environment.

    Parameters
    ----------
    backend : str
        ``"matplotlib"`` for MP4/GIF with MPPI sampled trajectories, or
        ``"plotly"`` for interactive HTML (trajectory + obstacles only).
    """
    import numpy as np

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
    animator.add_goal(desired_state[:2], radius=desired_state_radius)
    if obstacles and ellipsoids:
        animator.add_obstacles(obstacles, ellipsoid_radii=ellipsoids)
    animator.add_trajectory(x_idx=0, y_idx=1, label="Trajectory")
    animator.add_trajectory(
        x_idx=0,
        y_idx=1,
        data=estimates_np,
        style="scatter",
        label="Estimated Trajectory",
    )
    animator.show_time()

    # MPPI overlay (matplotlib only — too many traces for Plotly)
    if backend == "matplotlib":
        fig, ax = animator.build()

        sampled_lines = []
        for _ in range(mppi_args["plot_samples"]):
            (line,) = ax.plot([], [], "g", alpha=0.2)
            sampled_lines.append(line)
        (selected_line,) = ax.plot([], [], "b", linewidth=2, label="Selected Plan")

        sampled_key = (
            "sampled_x_traj"
            if "sampled_x_traj" in controller_data_keys
            else "robot_sampled_states"
        )
        sampled_idx = controller_data_keys.index(sampled_key)
        selected_idx = controller_data_keys.index("x_traj")
        state_dim = mppi_args["robot_state_dim"]

        def mppi_overlay(frame, _ax):
            if frame >= len(controller_data_items[sampled_idx]):
                return []
            sampled = controller_data_items[sampled_idx][frame]
            selected = controller_data_items[selected_idx][frame]

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
