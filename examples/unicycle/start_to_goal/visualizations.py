import matplotlib as mpl

mpl.use("Agg")  # headless render
mpl.style.use("fast")  # path-simplification on :contentReference[oaicite:0]{index=0}

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse


# --------------------------------------------------------------------------- #
#  STATIC PLOT (unchanged)
# --------------------------------------------------------------------------- #
def plot_trajectory(
    states,
    desired_state,
    desired_state_radius=0.25,
    obstacles=None,
    ellipsoids=None,
    x_lim=(-4, 4),
    y_lim=(-4, 4),
    title="System Behavior",
    fig=None,
    ax=None,
):
    obstacles = obstacles or []
    ellipsoids = ellipsoids or []

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.grid()

    #  ◾ static artists
    ax.plot(*desired_state, "ro", ms=5, label="desired_state")
    ax.add_patch(plt.Circle(desired_state, desired_state_radius, ec="r", fill=False, ls="--", lw=1))
    for obs, ell in zip(obstacles, ellipsoids):
        ax.add_patch(Ellipse(obs, 2 * ell[0], 2 * ell[1], fc="k"))

    ax.plot(states[:, 0], states[:, 1], label="Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig, ax


# --------------------------------------------------------------------------- #
#  ANIMATION  —  **O(N²) slice removed, blitting kept**                       #
# --------------------------------------------------------------------------- #
def animate(
    states,
    estimates,
    desired_state,
    desired_state_radius=0.1,
    obstacles=None,
    ellipsoids=None,
    x_lim=(-4, 4),
    y_lim=(-4, 4),
    dt=0.1,
    title="System Behavior",
    save_animation=True,
    animation_filename="system_behavior.gif",
):
    obstacles = obstacles or []
    ellipsoids = ellipsoids or []

    fig, ax = plt.subplots()
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.grid()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)

    #  ◾ static background
    ax.plot(*desired_state, "ro", ms=5)
    ax.add_patch(plt.Circle(desired_state, desired_state_radius, ec="r", fill=False, ls="--", lw=1))
    for obs, ell in zip(obstacles, ellipsoids):
        ax.add_patch(Ellipse(obs, 2 * ell[0], 2 * ell[1], fc="k"))

    #  ◾ dynamic artists
    (traj_line,) = ax.plot([], [], lw=1.5, label="Trajectory")
    (est_line,) = ax.plot([], [], lw=1.0, ls="--", label="Estimate")
    ax.legend(loc="best")

    #  ◾ incremental data containers — eliminates per-frame slicing
    xs, ys, exs, eys = [], [], [], []

    def init():
        traj_line.set_data([], [])
        est_line.set_data([], [])
        return traj_line, est_line

    def update(i):
        xs.append(states[i, 0])
        ys.append(states[i, 1])
        exs.append(estimates[i, 0])
        eys.append(estimates[i, 1])

        traj_line.set_data(xs, ys)
        est_line.set_data(exs, eys)
        return traj_line, est_line  # blit=True needs artists :contentReference[oaicite:1]{index=1}

    fps = int(round(1.0 / dt))
    ani = FuncAnimation(
        fig,
        update,
        frames=len(states),
        init_func=init,
        blit=True,
        interval=dt * 1000,
        cache_frame_data=False,
    )

    if save_animation:
        writer = FFMpegWriter(
            fps=fps, codec="libx264", extra_args=["-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        )
        # write .mp4 fast, convert to .gif if really needed
        mp4_name = animation_filename.rsplit(".", 1)[0] + ".mp4"
        ani.save(mp4_name, writer=writer)
        print(f"Saved animation to {mp4_name}")

    return fig, ax
