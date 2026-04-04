"""2D trajectory animation for fixed-wing UAV paths."""
from typing import List

import matplotlib.pyplot as plt
from jax import Array


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
    obstacles: List[Array],
    r_obs: List[List[float]],
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

    # Add ellipsoidal obstacles
    for obstacle, r_ob in zip(obstacles, r_obs):
        animator.add_obstacle(
            (float(obstacle[0]), float(obstacle[1])),
            ellipse_radii=(r_ob[0] / 2, r_ob[1] / 2),
            color="k",
        )

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
