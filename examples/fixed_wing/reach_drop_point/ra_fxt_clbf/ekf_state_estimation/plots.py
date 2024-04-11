import pickle
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.patches import Ellipse


pg = 0.75

file_path = "examples/fixed_wing/reach_drop_point/ra_fxt_clbf/ekf_state_estimation/results/"
file_name = file_path + f"ra_fxt_clf_pg{int(pg * 100)}.pkl"

# Load data from the pickle file
with open(file_name, "rb") as file:
    data = pickle.load(file)

# Assuming 'data' is a list or an array containing your 100 entries
dt = 1e-3
x = data["x"]
u = data["u"]
z = data["z"]

# Processing
high_idx = x.shape[0] - 340
t = np.arange(0.0, (high_idx * dt), dt)
colors = ["b", "r", "k", "c"]
lwidth = 3
fsize = 27

# Create a figure and axis
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.quit()

PLOT_XY = False
if PLOT_XY:
    _, ax_x = plt.subplots(figsize=(screen_width / 100, screen_height / 100))

    # plot goal region
    radius = 5  # You can adjust this value as needed
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    # plt.plot(x_circle, y_circle, label=None, color="g", linewidth=lwidth)
    ax_x.scatter(0.0, 0.0, color="g", s=100, marker="o", label="Goal Region")
    ax_x.scatter(500.0, 250.0, color="r", s=100, marker="x", label=r"$x_0$")

    ax_x.plot(x[:high_idx, 0], x[:high_idx, 1], linewidth=lwidth, color="b", label=r"$x$")
    ax_x.plot(z[:high_idx, 0], z[:high_idx, 1], linewidth=lwidth, color="r", label=r"$\hat x$")

    obstacles = [[0.0, 100.0, 0], [0.0, -100.0, 0], [350.0, 100.0, 0]]
    r_obs = [[200, 50, 1000], [1000, 50, 1000], [50, 100, 1000]]
    for obstacle, r_ob in zip(obstacles, r_obs):
        ax_x.add_patch(
            Ellipse(
                obstacle[:2],
                width=r_ob[0],
                height=r_ob[1],
                angle=0,
                edgecolor="k",
                facecolor="k",
                # fill=True,
                # linestyle="--",
                # linewidth=1,
            )
        )

    # Set the labels and title
    ax_x.set_xlabel(r"$x$", fontsize=fsize)
    ax_x.set_ylabel(r"$y$", fontsize=fsize)

    # plot limits
    ax_x.set_xlim([-100, 600])
    ax_x.set_ylim([-100, 300])

    # Add a legend
    ax_x.legend(fontsize=fsize - 5)

    for tick in ax_x.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_x.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    # Save the figure as a .png and .eps file
    plt.savefig(
        file_path + f"ra_fxt_clf_pg{int(pg * 100)}_xy_paths.png",
        format="png",
        bbox_inches="tight",
    )
    plt.savefig(
        file_path + f"ra_fxt_clf_pg{int(pg * 100)}_xy_paths.eps",
        format="eps",
        bbox_inches="tight",
    )


PLOT_YZ = False
if PLOT_YZ:
    _, ax_x = plt.subplots(figsize=(screen_width / 100, screen_height / 100))

    # plot goal region
    ax_x.scatter(0.0, 200.0, color="g", s=100, marker="o", label="Goal Region")
    ax_x.scatter(250.0, 250.0, color="r", s=100, marker="x", label=r"$x_0$")

    ax_x.plot(x[:high_idx, 1], x[:high_idx, 2], linewidth=lwidth, color="b", label=r"$x$")
    ax_x.plot(z[:high_idx, 1], z[:high_idx, 2], linewidth=lwidth, color="r", label=r"$\hat x$")

    obstacles = [[0.0, 100.0, 0], [0.0, -100.0, 0], [350.0, 100.0, 0]]
    r_obs = [[200, 50, 1000], [1000, 50, 1000], [50, 100, 1000]]
    for obstacle, r_ob in zip(obstacles, r_obs):
        ax_x.add_patch(
            Ellipse(
                obstacle[1:],
                width=r_ob[1],
                height=r_ob[2],
                angle=0,
                edgecolor="k",
                facecolor="k",
                # fill=True,
                # linestyle="--",
                # linewidth=1,
            )
        )

    # Set the labels and title
    ax_x.set_xlabel(r"$y$", fontsize=fsize)
    ax_x.set_ylabel(r"$z$", fontsize=fsize)

    # plot limits
    # ax_x.set_xlim([-100, 600])
    # ax_x.set_ylim([-100, 300])

    # Add a legend
    ax_x.legend(fontsize=fsize - 5)

    for tick in ax_x.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_x.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    # Save the figure as a .png and .eps file
    plt.savefig(
        file_path + f"ra_fxt_clf_pg{int(pg * 100)}_yz_paths.png",
        format="png",
        bbox_inches="tight",
    )
    plt.savefig(
        file_path + f"ra_fxt_clf_pg{int(pg * 100)}_yz_paths.eps",
        format="eps",
        bbox_inches="tight",
    )


PLOT_U = True
if PLOT_U:
    _, ax_u = plt.subplots(3, 1, figsize=(screen_width / 100, screen_height / 100))

    # plot goal region
    ax_u[0].plot(t, u[:high_idx, 0], linewidth=lwidth, color="b", label=r"$a$")
    ax_u[1].plot(t, u[:high_idx, 1], linewidth=lwidth, color="b", label=r"$\frac{g}{v}\tan \phi$")
    ax_u[2].plot(t, u[:high_idx, 2], linewidth=lwidth, color="b", label=r"$\omega$")

    # Set the labels and title
    ax_u[0].set_xlabel(r"$t$", fontsize=fsize)
    ax_u[0].set_ylabel(r"$u_1$", fontsize=fsize)
    ax_u[1].set_ylabel(r"$u_2$", fontsize=fsize)
    ax_u[2].set_ylabel(r"$u_3$", fontsize=fsize)

    # plot limits
    ax_u[0].set_ylim([-1, 1])
    ax_u[1].set_ylim([-20, 20])
    ax_u[2].set_ylim([-20, 20])

    # Add a legend
    ax_u[0].legend(fontsize=fsize - 5)
    ax_u[1].legend(fontsize=fsize - 5)
    ax_u[2].legend(fontsize=fsize - 5)

    for tick in ax_u[0].xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_u[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_u[1].xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_u[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_u[2].xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_u[2].yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    # Save the figure as a .png and .eps file
    plt.savefig(
        file_path + f"ra_fxt_clf_pg{int(pg * 100)}_controls.png",
        format="png",
        bbox_inches="tight",
    )
    plt.savefig(
        file_path + f"ra_fxt_clf_pg{int(pg * 100)}_controls.eps",
        format="eps",
        bbox_inches="tight",
    )


# Control Plot
PLOT_U = False
if PLOT_U:
    _, ax_u = plt.subplots(2, 1, figsize=(screen_width / 100, screen_height / 100))

    # Plot all 100 entries with a low alpha (translucent)
    for ii in range(100):
        ax_u[0].plot(t[: edits[ii]], new_u[ii, : edits[ii], 0], alpha=0.1)
        ax_u[1].plot(t[: edits[ii]], new_u[ii, : edits[ii], 1], alpha=0.1)
    for ii, cc in zip(selected_trials, colors):
        ax_u[0].plot(
            t[: edits[ii]],
            new_u[ii, : edits[ii], 0],
            label=rf"$u_1$ (sim. {ii+1})",
            color=cc,
            linewidth=lwidth,
        )
        ax_u[1].plot(
            t[: edits[ii]],
            new_u[ii, : edits[ii], 1],
            label=rf"$u_2$ (sim. {ii+1})",
            color=cc,
            linewidth=lwidth,
        )

    # Set the labels and title
    ax_u[1].set_xlabel(r"$t$ (sec)", fontsize=fsize)
    ax_u[0].set_ylabel(r"$u_1$", fontsize=fsize)
    ax_u[1].set_ylabel(r"$u_2$", fontsize=fsize)

    # plot limits
    xmax = np.max([np.max(np.abs(nu[:, 0])) for nu in new_u])
    ymax = np.max([np.max(np.abs(nu[:, 1])) for nu in new_u])
    ax_u[0].set_xlim([-0.03, 1.0])
    ax_u[1].set_xlim([-0.03, 1.0])
    ax_u[0].set_ylim([-20, 20])
    ax_u[1].set_ylim([-20, 20])

    # Add a legend
    ax_u[0].legend(fontsize=fsize - 5)
    ax_u[1].legend(fontsize=fsize - 5)

    for tick in ax_u[0].xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_u[1].xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)  # Set the desired font size

    for tick in ax_u[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_u[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    # Save the figure as a .png and .eps file
    plt.savefig(
        file_path + f"ra_fxt_clf_n{n_trials}_pg{int(pg * 100)}_controls.png",
        format="png",
        bbox_inches="tight",
    )
    plt.savefig(
        file_path + f"ra_fxt_clf_n{n_trials}_pg{int(pg * 100)}_controls.eps",
        format="eps",
        bbox_inches="tight",
    )

PLOT_V = False
if PLOT_V:
    _, ax_v = plt.subplots(figsize=(screen_width / 100, screen_height / 100))

    v_func = 0.5 * (new_x[:, :, 0] ** 2 + new_x[:, :, 1] ** 2 - (0.05**2))

    # Plot all 100 entries with a low alpha (translucent)
    ax_v.plot(t, v_func.T, alpha=0.1)
    for ii, cc in zip(selected_trials, colors):
        ax_v.plot(
            t,
            v_func[ii, :],
            label=f"(sim. {ii+1})",
            color=cc,
            linewidth=lwidth,
        )

    # Set the labels and title
    ax_v.set_xlabel(r"$t$ (sec)", fontsize=fsize)
    ax_v.set_ylabel(r"$V$", fontsize=fsize)

    # plot limits
    ax_v.set_xlim([-0.03, 1.0])
    ax_v.set_ylim([-0.1, 2.0])

    # Add a legend
    ax_v.legend(fontsize=fsize - 5)

    for tick in ax_v.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_v.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    # Save the figure as a .png and .eps file
    plt.savefig(
        file_path + f"ra_fxt_clf_n{n_trials}_pg{int(pg * 100)}_lyapunov_vals.png",
        format="png",
        bbox_inches="tight",
    )
    plt.savefig(
        file_path + f"ra_fxt_clf_n{n_trials}_pg{int(pg * 100)}_lyapunov_vals.eps",
        format="eps",
        bbox_inches="tight",
    )


# Show the plot
plt.show()
