import matplotlib

# matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from typing import List
from jax import Array


#! PLOTTING
def plot_trajectory(
    states,
    title=None,
):
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
    desired_state,
    desired_state_radius,
    x_lim=(-4, 4),
    y_lim=(-4, 4),
    dt=0.1,
    title="System Behavior",
    save_animation=False,
    animation_filename="system_behavior.gif",
    num_robots=2,
):
    def init():
        for i in range(num_robots):
            trajectory[i].set_data([], [])
        return trajectory

    def update(frame):
        for i in range(num_robots):
            trajectory[i].set_data(states[:frame, 2 * i], states[:frame, 2 * i + 1])
        _ = states[frame]
        _ = estimates[frame]
        return trajectory

    fig, ax = plt.subplots()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    desired_state_radius = 0.1
    # for i in range(num_robots):
    #     ax.plot(
    #         desired_state[2 * i],
    #         desired_state[2 * i + 1],
    #         "ro",
    #         markersize=5,
    #         label="desired_state " + str(i + 1),
    #     )
    #     ax.add_patch(
    #         plt.Circle(
    #             desired_state[2 * i : 2 * i + 2],
    #             desired_state_radius,
    #             color="r",
    #             fill=False,
    #             linestyle="--",
    #             linewidth=1,
    #         )
    #     )

    from matplotlib import animation

    trajectory = [0] * num_robots
    for i in range(num_robots):
        (trajectory[i],) = ax.plot([], [])  # , label="Trajectory - robot " + str(i + 1))

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid()

    ani = FuncAnimation(
        fig, update, frames=len(states), init_func=init, blit=True, interval=dt * 100
    )

    if save_animation:
        ani.save(animation_filename, writer="imagemagick", fps=15)

    plt.show()

    return fig, ax
