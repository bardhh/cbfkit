import matplotlib

# matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

    def update(frame):
        trajectory.set_data(states[:frame, 0], states[:frame, 1])
        etrajectory.set_data(estimates[:frame, 0], estimates[:frame, 1])
        _, _, _ = states[frame]
        _, _, _ = estimates[frame]
        return (
            trajectory,
            etrajectory,
        )

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

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid()

    ani = FuncAnimation(
        fig, update, frames=len(states), init_func=init, blit=True, interval=dt * 100
    )

    if save_animation:
        ani.save(animation_filename, writer="imagemagick", fps=15)

    plt.show()

    return fig, ax
