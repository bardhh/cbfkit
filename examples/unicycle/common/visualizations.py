"""Trajectory plotting and animation for unicycle simulations."""
# matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


#! PLOTTING
def plot_trajectory(
    # fig,
    # ax,
    states,
    desired_state,
    desired_state_radius=0.25,
    obstacles=[],
    ellipsoids=[],
    x_lim=(-4, 4),
    y_lim=(-4, 4),
    title="System Behavior",
):
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
    ax.legend()
    ax.grid()

    # plt.show()

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
    from cbfkit.utils.animator import CBFAnimator

    animator = CBFAnimator(states, dt=dt, x_lim=x_lim, y_lim=y_lim, title=title)
    animator.add_goal(desired_state[:2], radius=desired_state_radius)
    if obstacles and ellipsoids:
        animator.add_obstacles(obstacles, ellipsoid_radii=ellipsoids)
    # Draw estimate as scatter points under the true trajectory
    animator.add_trajectory(
        x_idx=0, y_idx=1, data=estimates,
        color="orange", label="Estimated Trajectory", style="scatter",
        alpha=0.5, zorder=2,
    )
    animator.add_trajectory(
        x_idx=0, y_idx=1,
        color="blue", label="Trajectory",
        zorder=3,
    )

    if save_animation:
        animator.save(animation_filename)

    animator.show()
    return animator.fig, animator.ax
