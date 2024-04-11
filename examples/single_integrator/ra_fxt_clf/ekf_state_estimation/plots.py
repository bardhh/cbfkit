import pickle
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

n_trials = 100
pg = 0.55

file_path = "examples/single_integrator/ra_fxt_clf/ekf_state_estimation/results/"
file_name = file_path + f"ra_fxt_clf_n{n_trials}_pg{int(pg * 100)}.pkl"

# Load data from the pickle file
with open(file_name, "rb") as file:
    data = pickle.load(file)

# Assuming 'data' is a list or an array containing your 100 entries
dt = 1e-3
x = data["x"]
u = data["u"]
z = data["z"]

# Processing
high_idx = np.max([xx.shape[0] for xx in x])
t = np.arange(0.0, (high_idx * dt), dt)
new_x = np.zeros((len(x), high_idx, 2))
new_u = np.zeros((len(x), high_idx, 2))
new_z = np.zeros((len(x), high_idx, 2))
edits = []
for ii, (xx, uu, zz) in enumerate(zip(x, u, z)):
    diff = high_idx - xx.shape[0]
    edits.append(xx.shape[0])
    # Create an array with the final value of the second array
    additional_entries_x = np.tile(xx[-1], (diff, 1))
    additional_entries_u = np.tile(uu[-1], (diff, 1))
    additional_entries_z = np.tile(zz[-1], (diff, 1))

    # Concatenate the additional entries to the second array
    new_x[ii] = np.vstack((xx, additional_entries_x))
    new_u[ii] = np.vstack((uu, additional_entries_u))
    new_z[ii] = np.vstack((zz, additional_entries_z))


selected_trials = [0, 24, 49, 89]
colors = ["b", "r", "k", "c"]
lwidth = 3
fsize = 27

# Create a figure and axis
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.quit()

PLOT_X = False
if PLOT_X:
    _, ax_x = plt.subplots(figsize=(screen_width / 100, screen_height / 100))

    # plot goal region
    radius = 0.05  # You can adjust this value as needed
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    plt.plot(x_circle, y_circle, label=None, color="g", linewidth=lwidth)
    ax_x.scatter(0.0, 0.0, color="g", s=100, marker="o", label="Goal Region")

    # Plot all 100 entries with a low alpha (translucent)
    ax_x.plot(new_x[:, :, 0], new_x[:, :, 1], alpha=0.01)
    for ii, cc in zip(selected_trials, colors):
        ax_x.scatter(new_x[ii, 0, 0], new_x[ii, 0, 1], color=cc, s=300, marker="x")
        ax_x.plot(
            new_x[ii, :, 0],
            new_x[ii, :, 1],
            label=rf"$x$ (sim. {ii+1})",
            color=cc,
            linewidth=lwidth,
        )
        ax_x.plot(
            new_z[ii, :, 0],
            new_z[ii, :, 1],
            ":",
            label=rf"$\hat x$ (sim. {ii+1})",
            color=cc,
            linewidth=lwidth,
        )

    # Set the labels and title
    ax_x.set_xlabel(r"$x_1$", fontsize=fsize)
    ax_x.set_ylabel(r"$x_2$", fontsize=fsize)
    # ax_x.set_title("States and Estimates", fontsize=fsize + 5)

    # plot limits
    xmax = np.max([np.max(np.abs(nx[:, 0])) for nx in new_x])
    ymax = np.max([np.max(np.abs(nx[:, 1])) for nx in new_x])
    ax_x.set_xlim([-xmax, xmax])
    ax_x.set_ylim([-ymax, ymax])

    # Add a legend
    ax_x.legend(fontsize=fsize - 5)

    for tick in ax_x.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    for tick in ax_x.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize - 5)

    # Save the figure as a .png and .eps file
    plt.savefig(
        file_path + f"ra_fxt_clf_n{n_trials}_pg{int(pg * 100)}_state_paths.png",
        format="png",
        bbox_inches="tight",
    )
    plt.savefig(
        file_path + f"ra_fxt_clf_n{n_trials}_pg{int(pg * 100)}_state_paths.eps",
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

PLOT_V = True
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
