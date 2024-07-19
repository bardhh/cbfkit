import matplotlib

# matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Ellipse
from typing import List
from jax import Array
import numpy as np
import jax.numpy as jnp


class circle:

    def __init__(self, ax, pos=np.array([0, 0]), radius=1.0):
        self.X = pos
        self.radius = radius
        self.id = id
        self.type = "circle"

        self.render(ax)

    def render(self, ax):
        circ = plt.Circle(
            (self.X[0], self.X[1]), self.radius, linewidth=1, edgecolor="k", facecolor="k"
        )
        ax.add_patch(circ)


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
):
    def init(trajectory):
        trajectory[0].set_data([], [])
        plot_num_samples = mppi_args["plot_samples"]
        for i in range(plot_num_samples):  # sampled trajectories
            trajectory[i + 1].set_data([], [])
        trajectory[-1].set_data([], [])
        return trajectory

    def update(frame, trajectory, text):
        # Robot trajectory until current time
        text.set_text(f"time: {round(dt * frame,2)}")
        trajectory[0].set_data(states[:frame, 0], states[:frame, 1])
        _ = states[frame]
        _ = estimates[frame]
        robot_sampled_states = controller_data_items[frame][
            controller_data_keys.index("robot_sampled_states")
        ]
        robot_selected_states = controller_data_items[frame][controller_data_keys.index("x_traj")]

        # Sampled Trajectories
        for i in range(mppi_args["plot_samples"]):
            trajectory[i + 1].set_data(
                robot_sampled_states[mppi_args["robot_state_dim"] * i, :],
                robot_sampled_states[mppi_args["robot_state_dim"] * i + 1, :],
            )

        # Selected Trajectory
        trajectory[-1].set_data(robot_selected_states[0, :], robot_selected_states[1, :])

        return trajectory, text

    plt.ion()
    fig, ax = plt.subplots()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # desired_state_radius = 0.1

    for k in range(len(desired_state)):
        ax.add_patch(
            plt.Circle(
                desired_state[k],
                desired_state_radius,
                color="r",
                fill=False,
                linestyle="--",
                linewidth=1,
            )
        )

    if obstacle != None:
        circle(ax, pos=obstacle, radius=obstacle_radius)

    from matplotlib import animation

    text = ax.text(1, 9, f"time: {dt}")
    trajectory = [0] * (1 + mppi_args["plot_samples"] + 1)
    (trajectory[0],) = ax.plot([], [], "k")
    for i in range(mppi_args["plot_samples"]):
        (trajectory[i + 1],) = ax.plot([], [], "g", alpha=0.2)
    (trajectory[-1],) = ax.plot([], [], "b")

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid()

    trajectory = init(trajectory)

    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=int(1 / dt), metadata=metadata)
    with writer.saving(fig, animation_filename + ".mp4", 100):
        for i in range(len(states)):
            trajectory, text = update(i, trajectory, text)
            fig.canvas.draw()
            fig.canvas.flush_events()
            writer.grab_frame()

    plt.ioff()
    plt.savefig(animation_filename + ".eps")
    plt.show()

    return fig, ax
