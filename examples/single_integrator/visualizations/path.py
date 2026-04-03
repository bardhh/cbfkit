"""Trajectory plotting and animation for single integrator examples."""
import matplotlib.pyplot as plt


def plot_trajectory(states, title=None):
    """Plot state components (X, Y, Z) over time."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    lbl = ["X", "Y", "Z"]
    for ii in range(3):
        axs[ii].plot(states[:, ii], label=lbl[ii])
        axs[ii].set_xlabel("Step")
        axs[ii].set_ylabel(lbl[ii])
        axs[ii].legend()
        axs[ii].grid()

    if title:
        fig.suptitle(title)

    return fig, axs


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
        x_idx=0,
        y_idx=1,
        data=estimates,
        color="tab:orange",
        label="Estimated Trajectory",
    )
    animator.add_trajectory(
        x_idx=0,
        y_idx=1,
        color="tab:blue",
        label="Trajectory",
    )

    if save_animation:
        animator.save(animation_filename)

    animator.show()
    return animator.fig, animator.ax
