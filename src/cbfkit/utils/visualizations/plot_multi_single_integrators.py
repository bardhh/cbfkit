import matplotlib.pyplot as plt

from cbfkit.utils.animator import CBFAnimator


#! PLOTTING
def plot_trajectory(
    states,
    title=None,
):
    """Plot the trajectory of multiple single integrators."""
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
    """Animate the system behavior for multiple single integrators."""
    animator = CBFAnimator(states, dt=dt, x_lim=x_lim, y_lim=y_lim, title=title)

    for i in range(num_robots):
        animator.add_trajectory(x_idx=2 * i, y_idx=2 * i + 1)

    if save_animation:
        animator.save(animation_filename)
    else:
        animator.show()

    return animator.fig, animator.ax
