"""Trajectory plotting and animation for Van der Pol oscillator paths."""
import matplotlib.pyplot as plt


#! PLOTTING
def plot_trajectory(
    states,
    title=None,
):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    lbl = ["X", "Y", "Z"]
    for ii in range(3):
        axs[ii].plot(states[:, ii], label=lbl[ii])

    # ax.set_xlim(x_lim)
    # ax.set_ylim(y_lim)

    # ax.plot(desired_state[0], desired_state[1], "ro", markersize=5, label="desired_state")
    # ax.add_patch(
    #     plt.Circle(
    #         desired_state,
    #         desired_state_radius,
    #         color="r",
    #         fill=False,
    #         linestyle="--",
    #         linewidth=1,
    #     )
    # )
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

    # ax.plot(states[:, 0], states[:, 1], label="Trajectory")

    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    # axs.set_title(title)
    # axs.legend(loc="upper left")
    # axs.grid()

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
):
    from cbfkit.utils.animator import CBFAnimator

    animator = CBFAnimator(states, dt=dt, x_lim=x_lim, y_lim=y_lim, title=title)
    animator.add_goal(desired_state[:2], radius=desired_state_radius)
    animator.add_trajectory(
        x_idx=0, y_idx=1, data=estimates,
        color="tab:orange", label="Estimated Trajectory",
    )
    animator.add_trajectory(
        x_idx=0, y_idx=1,
        color="tab:blue", label="Trajectory",
    )

    if save_animation:
        animator.save(animation_filename)

    animator.show()
    return animator.fig, animator.ax
