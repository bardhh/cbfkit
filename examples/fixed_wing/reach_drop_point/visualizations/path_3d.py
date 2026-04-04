"""3D trajectory plotting and animation for fixed-wing UAV paths."""
from typing import List

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from matplotlib.animation import FuncAnimation


def plot(
    trajectory: Array,
    obstacles: List[Array],
    r_obs: List[float],
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

    for obstacle, r in zip(obstacles, r_obs):
        theta = jnp.linspace(0, 2 * jnp.pi, 50)
        phi = jnp.linspace(0, jnp.pi, 50)
        theta, phi = jnp.meshgrid(theta, phi)

        x_sphere = r * jnp.sin(phi) * jnp.cos(theta) + obstacle[0]
        y_sphere = r * jnp.sin(phi) * jnp.sin(theta) + obstacle[1]
        z_sphere = r * jnp.cos(phi) + obstacle[2]

        ax.plot_surface(x_sphere, y_sphere, z_sphere, color="r", alpha=0.6)  # type: ignore[attr-defined]

    return fig, ax


def animate(
    trajectory: Array,
    obstacles: List[Array],
    r_obs: List[List[float]],
    dt: float = 0.01,
    save_animation=False,
    animation_filename="3d_animation.gif",
):
    speedup = 7
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    (traj,) = ax.plot(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2])

    for obstacle, r in zip(obstacles, r_obs):
        theta, phi = jnp.mgrid[0 : 2 * jnp.pi : 20j, 0 : jnp.pi : 10j]  # type: ignore[misc]

        x_ellipse = r[0] * jnp.sin(phi) * jnp.cos(theta) + obstacle[0]
        y_ellipse = r[1] * jnp.sin(phi) * jnp.sin(theta) + obstacle[1]
        z_ellipse = r[2] * jnp.cos(phi) + obstacle[2]

        ax.plot_surface(x_ellipse, y_ellipse, z_ellipse, color="r", alpha=0.4)  # type: ignore[attr-defined]

    def update(num):
        num = int(num * speedup)
        traj.set_data(trajectory[:num, 0], trajectory[:num, 1])
        traj.set_3d_properties(trajectory[:num, 2])
        return (traj,)

    ani = FuncAnimation(fig, update, frames=int(len(trajectory) / speedup), blit=True)

    if save_animation:
        from cbfkit.utils.animator import save_animation as _save_animation

        _save_animation(ani, animation_filename)

    return fig, ax
