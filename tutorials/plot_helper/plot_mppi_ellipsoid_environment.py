import matplotlib

# matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
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
    # for x, y, r in zip(CX, CY, R):
    #     ax.add_patch(
    #         plt.Circle(
    #             (x, y),
    #             r,
    #             color="k",
    #             fill=True,
    #             linestyle="-",
    #             linewidth=1,
    #         )
    #     )
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
    # ax.legend()
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
):
    def init():
        trajectory.set_data([], [])
        etrajectory.set_data([], [])
        return (trajectory,)

    def update(frame, trajectory, etrajectory, mppi_sampled_trajectory, mppi_selected_trajectory):
        trajectory.set_data(states[:frame, 0], states[:frame, 1])
        etrajectory.set_data(estimates[:frame, 0], estimates[:frame, 1])
        _, _, _, _ = states[frame]
        _, _, _, _ = estimates[frame]

        robot_sampled_states = controller_data_items[frame][
            controller_data_keys.index("robot_sampled_states")
        ]
        robot_selected_states = controller_data_items[frame][controller_data_keys.index("x_traj")]

        # Sampled Trajectories
        for i in range(mppi_args["plot_samples"]):
            mppi_sampled_trajectory[i].set_data(
                robot_sampled_states[mppi_args["robot_state_dim"] * i, :],
                robot_sampled_states[mppi_args["robot_state_dim"] * i + 1, :],
            )

        # Selected Trajectory
        mppi_selected_trajectory.set_data(robot_selected_states[0, :], robot_selected_states[1, :])

        return trajectory, etrajectory, mppi_sampled_trajectory, mppi_selected_trajectory

    plt.ion()
    fig, ax = plt.subplots()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    desired_state_radius = 0.1
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

    (trajectory,) = ax.plot([], [], label="Trajectory")
    (etrajectory,) = ax.plot([], [], label="Estimated Trajectory")

    mppi_sampled_trajectory = [0] * (mppi_args["plot_samples"])
    for i in range(mppi_args["plot_samples"]):
        (mppi_sampled_trajectory[i],) = ax.plot([], [], "g", alpha=0.2)
    (mppi_selected_trajectory,) = ax.plot([], [], "b")

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid()

    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=int(1 / dt), metadata=metadata)
    print(animation_filename)
    with writer.saving(fig, animation_filename + ".mp4", 100):
        for i in range(len(states)):
            trajectory, etrajectory, mppi_sampled_trajectory, mppi_selected_trajectory = update(
                i, trajectory, etrajectory, mppi_sampled_trajectory, mppi_selected_trajectory
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            writer.grab_frame()

    plt.ioff()
    plt.savefig(animation_filename + ".eps")
    plt.show()

    # ani = FuncAnimation(
    #     fig, update, frames=len(states), init_func=init, blit=True, interval=dt * 100
    # )

    # if save_animation:
    #     ani.save(animation_filename, writer="imagemagick", fps=15)

    # plt.show()

    return fig, ax
