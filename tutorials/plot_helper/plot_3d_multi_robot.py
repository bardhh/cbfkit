import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


def plot_ellipse_3d(ax, center, radii, rotation, color="blue", alpha=0.2):
    """
    Plots a 3D ellipse on the provided axes.

    Parameters:
    - ax: the 3D axis to plot the ellipse on.
    - center: the center of the ellipse [x, y, z].
    - radii: the semi-axis lengths of the ellipse [rx, ry, rz].
    - rotation: a 3x3 rotation matrix to orient the ellipse.
    - color: the color of the ellipse.
    - alpha: transparency of the ellipse.
    """
    # Generate data for a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate and translate data
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(rotation, [x[i, j], y[i, j], z[i, j]]) + center

    ax.plot_surface(x, y, z, color=color, rstride=4, cstride=4, alpha=alpha)


def point_to_ellipsoid_distance(p, c, r, R):
    """
    Compute the distance from point p to an ellipsoid centered at c, with radii r and rotation R.
    """
    # Direction vector from c to p
    u = p - c
    norm_u = np.linalg.norm(u)
    if norm_u == 0:
        # p coincides with c, so it is inside the ellipsoid
        return -np.min(r)  # Return negative distance using minimum radius

    u_normalized = u / norm_u

    # Compute K
    inv_r_squared = np.diag(1.0 / (r**2))
    K = u_normalized.T @ R @ inv_r_squared @ R.T @ u_normalized

    # Compute s
    s = 1.0 / np.sqrt(K)

    # Closest point on the ellipsoid
    x = c + s * u_normalized

    # Distance from p to x
    d = np.linalg.norm(p - x)

    # If s >= 1, p is outside the ellipsoid
    if s >= 1:
        return d
    else:
        # p is inside the ellipsoid
        return -d


def animate_3d(
    states,
    desired_state,
    desired_state_radius,
    x_lim=(-5, 5),
    y_lim=(-5, 5),
    z_lim=(-5, 5),
    dt=0.1,
    title="System Behavior",
    save_animation=False,
    animation_filename="system_behavior.mp4",
):
    # Compute time vector
    time = np.arange(0, len(states) * dt, dt)

    # Create a figure with GridSpec for two subplots
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[2, 1])

    # Create 3D trajectory subplot
    ax_traj = fig.add_subplot(gs[0], projection="3d")

    # Set the axes limits
    ax_traj.set_xlim(x_lim)
    ax_traj.set_ylim(y_lim)
    ax_traj.set_zlim(z_lim)

    # Labels and title
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.set_zlabel("Z [m]")
    ax_traj.set_title(title)

    # Desired state color
    desired_state_color = "red"

    # Plot the desired state
    ax_traj.scatter(
        desired_state[0],
        desired_state[1],
        desired_state[2],
        color=desired_state_color,
        s=50,
        label="Desired State",
    )

    # Optionally, plot a sphere around the desired state to represent the desired_state_radius
    # Create data for a sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = desired_state_radius * np.outer(np.cos(u), np.sin(v)) + desired_state[0]
    y_sphere = desired_state_radius * np.outer(np.sin(u), np.sin(v)) + desired_state[1]
    z_sphere = desired_state_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + desired_state[2]
    ax_traj.plot_wireframe(
        x_sphere, y_sphere, z_sphere, color=desired_state_color, linewidth=0.5, alpha=0.3
    )

    # Initialize the trajectory line with the same color as the desired state
    (line_traj,) = ax_traj.plot([], [], [], lw=2, label="Trajectory", color=desired_state_color)

    # Create distance subplot
    ax_dist = fig.add_subplot(gs[1])
    ax_dist.set_xlim(0, time[-1])
    max_distance = np.max(np.linalg.norm(states - desired_state, axis=1))
    ax_dist.set_ylim(0, max_distance * 1.1)
    ax_dist.set_xlabel("Time [s]")
    ax_dist.set_ylabel("Distance [m]")
    ax_dist.set_title("Distance to Desired State")
    (line_dist,) = ax_dist.plot([], [], lw=2, color=desired_state_color)
    ax_dist.grid(True)

    # Initialize function
    def init():
        line_traj.set_data([], [])
        line_traj.set_3d_properties([])
        line_dist.set_data([], [])
        return line_traj, line_dist

    # Update function for animation
    def update(num):
        # Update trajectory line
        x = states[:num, 0]
        y = states[:num, 1]
        z = states[:num, 2]
        line_traj.set_data(x, y)
        line_traj.set_3d_properties(z)

        # Update distance line
        distances = np.linalg.norm(states[:num, :] - desired_state, axis=1)
        line_dist.set_data(time[:num], distances)

        return line_traj, line_dist

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(states), init_func=init, blit=True, interval=dt * 1000
    )

    # Add legend and grid
    # ax_traj.legend()
    ax_traj.grid(True)

    # Save the animation if required
    if save_animation:
        writer = FFMpegWriter(fps=15)
        ani.save(animation_filename, writer=writer)

    # Show the plot
    plt.tight_layout()
    plt.show()

    return fig, (ax_traj, ax_dist)


def animate_3d_multi_robot(
    states,
    desired_states,
    desired_state_radius,
    num_robots,
    ellipse_centers=None,  # List of ellipse centers for obstacles
    ellipse_radii=None,  # List of ellipse radii for obstacles
    ellipse_rotations=None,  # List of rotation matrices for obstacles
    x_lim=(-5, 5),
    y_lim=(-5, 5),
    z_lim=(-5, 5),
    dt=0.1,
    state_dimension_per_robot=3,
    title="Multi-Robot Trajectory",
    save_animation=False,
    animation_filename="system_behavior.mp4",
    include_min_distance_plot=False,  # Control inclusion of min distance between robots
    include_min_distance_to_obstacles_plot=False,  # Control inclusion of min distance to obstacles
    threshold=None,
):
    # Compute time vector
    time = np.arange(0, len(states) * dt, dt)

    # Calculate the number of subplots based on whether we include the min distance plots
    num_subplots = 2  # Trajectory and Distance to Goal plots
    if include_min_distance_plot:
        num_subplots += 1
    if include_min_distance_to_obstacles_plot:
        num_subplots += 1

    # Set up figure and GridSpec based on the number of subplots
    if num_subplots == 2:
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(1, 2, width_ratios=[2, 1])
    elif num_subplots == 3:
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, width_ratios=[2, 1, 1])
    elif num_subplots == 4:
        fig = plt.figure(figsize=(22, 6))
        gs = GridSpec(1, 4, width_ratios=[2, 1, 1, 1])

    # Create 3D trajectory subplot
    ax_traj = fig.add_subplot(gs[0], projection="3d")

    # Set the axes limits
    ax_traj.set_xlim(x_lim)
    ax_traj.set_ylim(y_lim)
    ax_traj.set_zlim(z_lim)

    # Labels and title
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.set_zlabel("Z [m]")
    ax_traj.set_title(title)

    # Get a list of colors
    colors = plt.cm.get_cmap("tab10", num_robots).colors

    # Initialize lists for lines
    lines_traj = []
    lines_goal_dist = []
    lines_min_dist = []
    lines_obstacle_dist = []

    # Create distance to goal subplot
    ax_goal_dist = fig.add_subplot(gs[1])
    ax_goal_dist.set_xlim(0, time[-1])
    ax_goal_dist.set_xlabel("Time [s]")
    ax_goal_dist.set_ylabel("Distance to Goal [m]")
    ax_goal_dist.set_title("Distance to Desired States")
    ax_goal_dist.grid(True)

    max_goal_distance = 0  # To adjust y-axis limit for distance to goal plot

    # Plot the desired states and initialize lines
    for i in range(num_robots):
        idx = state_dimension_per_robot * i
        color = colors[i % num_robots]  # Ensure each robot gets a unique color

        # Plot desired state
        ax_traj.scatter(
            desired_states[idx],
            desired_states[idx + 1],
            desired_states[idx + 2],
            color=color,
            s=50,
            label=f"Desired State {i+1}",
        )

        # Initialize trajectory line
        (line_traj,) = ax_traj.plot([], [], [], lw=2, label=f"Trajectory {i+1}", color=color)
        lines_traj.append(line_traj)

        # Initialize distance to goal line
        (line_goal_dist,) = ax_goal_dist.plot([], [], lw=2, label=f"Robot {i+1}", color=color)
        lines_goal_dist.append(line_goal_dist)

        # Calculate maximum distance for y-axis limit
        goal_distances = np.linalg.norm(
            states[:, idx : idx + 3] - desired_states[idx : idx + 3], axis=1
        )
        max_goal_distance = max(max_goal_distance, np.max(goal_distances))

    # Adjust y-axis limit for distance to goal plot
    ax_goal_dist.set_ylim(0, max_goal_distance * 1.1)

    # Plot the elliptic obstacles
    if ellipse_centers is not None and ellipse_radii is not None and ellipse_rotations is not None:
        for i in range(len(ellipse_centers)):
            plot_ellipse_3d(
                ax_traj,
                ellipse_centers[i],
                ellipse_radii[i],
                ellipse_rotations[i],
                color="black",
                alpha=0.2,
            )

    # If including minimum distance between robots plot, initialize it
    if include_min_distance_plot:
        # Create minimum distance subplot
        ax_min_dist = fig.add_subplot(gs[2])
        ax_min_dist.set_xlim(0, time[-1])
        ax_min_dist.set_xlabel("Time [s]")
        ax_min_dist.set_ylabel("Minimum Distance [m]")
        ax_min_dist.set_title("Minimum Distances Between Robots")
        ax_min_dist.grid(True)

        # Initialize distance lines for each robot
        min_distances = np.zeros((len(states), num_robots))
        max_min_distance = 0  # To adjust y-axis limit for minimum distance plot

        # Compute minimum distances between robots
        for t in range(len(states)):
            positions = states[t, :].reshape(num_robots, state_dimension_per_robot)
            for i in range(num_robots):
                # Compute distances to other robots
                diffs = positions - positions[i]
                dists = np.linalg.norm(diffs, axis=1)
                # Exclude the distance to itself
                dists[i] = np.inf
                min_distances[t, i] = np.min(dists)

        max_min_distance = np.max(min_distances)

        # Adjust y-axis limit for minimum distance plot
        ax_min_dist.set_ylim(0, max_min_distance * 1.1)

        # Plot the threshold line if provided
        if threshold is not None:
            ax_min_dist.axhline(y=threshold, color="red", linestyle="--", label="Threshold")

        # Initialize lines for minimum distances
        for i in range(num_robots):
            color = colors[i % num_robots]
            (line_min_dist,) = ax_min_dist.plot([], [], lw=2, label=f"Robot {i+1}", color=color)
            lines_min_dist.append(line_min_dist)

        # ax_min_dist.legend()  # Add legend to display the threshold line

    # If including minimum distance to obstacles plot, initialize it
    if include_min_distance_to_obstacles_plot:
        if ellipse_centers is None or ellipse_radii is None or ellipse_rotations is None:
            raise ValueError(
                "To compute distances to obstacles, ellipse_centers, ellipse_radii, and ellipse_rotations must be provided."
            )

        # Create obstacle distance subplot
        subplot_index = 2 if not include_min_distance_plot else 3
        ax_obstacle_dist = fig.add_subplot(gs[subplot_index])
        ax_obstacle_dist.set_xlim(0, time[-1])
        ax_obstacle_dist.set_xlabel("Time [s]")
        ax_obstacle_dist.set_ylabel("Distance to Obstacles [m]")
        ax_obstacle_dist.set_title("Minimum Distance to Obstacles")
        ax_obstacle_dist.grid(True)

        # Initialize obstacle distance lines for each robot
        obstacle_distances = np.zeros((len(states), num_robots))
        max_obstacle_distance = 0  # To adjust y-axis limit for obstacle distance plot

        # Compute minimum distances to obstacles
        num_obstacles = len(ellipse_centers)
        for t in range(len(states)):
            for i in range(num_robots):
                idx = state_dimension_per_robot * i
                p = states[t, idx : idx + 3]
                min_distance = np.inf
                for j in range(num_obstacles):
                    c = ellipse_centers[j]
                    r = ellipse_radii[j]
                    R = ellipse_rotations[j]
                    d = point_to_ellipsoid_distance(p, c, r, R)
                    if d < min_distance:
                        min_distance = d
                obstacle_distances[t, i] = min_distance

        max_obstacle_distance = np.max(obstacle_distances)
        min_obstacle_distance = np.min(obstacle_distances)

        # Adjust y-axis limit for obstacle distance plot
        ax_obstacle_dist.set_ylim(0, max_obstacle_distance * 1.1)

        # Initialize lines for obstacle distances
        for i in range(num_robots):
            color = colors[i % num_robots]
            (line_obstacle_dist,) = ax_obstacle_dist.plot(
                [], [], lw=2, label=f"Robot {i+1}", color=color
            )
            lines_obstacle_dist.append(line_obstacle_dist)

        # ax_obstacle_dist.legend()

    # Initialize function
    def init():
        artists = []
        for line_traj in lines_traj:
            line_traj.set_data([], [])
            line_traj.set_3d_properties([])
            artists.append(line_traj)
        for line_goal_dist in lines_goal_dist:
            line_goal_dist.set_data([], [])
            artists.append(line_goal_dist)
        if include_min_distance_plot:
            for line_min_dist in lines_min_dist:
                line_min_dist.set_data([], [])
                artists.append(line_min_dist)
        if include_min_distance_to_obstacles_plot:
            for line_obstacle_dist in lines_obstacle_dist:
                line_obstacle_dist.set_data([], [])
                artists.append(line_obstacle_dist)
        return artists

    # Update function for animation
    def update(num):
        artist_list = []
        for i, (line_traj, line_goal_dist) in enumerate(zip(lines_traj, lines_goal_dist)):
            idx = state_dimension_per_robot * i
            x = states[:num, idx]
            y = states[:num, idx + 1]
            z = states[:num, idx + 2]
            line_traj.set_data(x, y)
            line_traj.set_3d_properties(z)
            artist_list.append(line_traj)

            # Update distance to goal line
            distances_to_goal = np.linalg.norm(
                states[:num, idx : idx + 3] - desired_states[idx : idx + 3], axis=1
            )
            line_goal_dist.set_data(time[:num], distances_to_goal)
            artist_list.append(line_goal_dist)

        if include_min_distance_plot:
            for i, line_min_dist in enumerate(lines_min_dist):
                line_min_dist.set_data(time[:num], min_distances[:num, i])
                artist_list.append(line_min_dist)

        if include_min_distance_to_obstacles_plot:
            for i, line_obstacle_dist in enumerate(lines_obstacle_dist):
                line_obstacle_dist.set_data(time[:num], obstacle_distances[:num, i])
                artist_list.append(line_obstacle_dist)

        return artist_list

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(states), init_func=init, blit=True, interval=dt * 1000
    )

    # Add legend and grid
    ax_traj.grid(True)

    # ax_obstacle_dist.legend()

    # Save the animation if required
    if save_animation:
        print("plotting...")
        if animation_filename.endswith(".gif"):
            ani.save(animation_filename, writer="imagemagick", fps=15)
        else:
            writer = FFMpegWriter(fps=15)
            ani.save(animation_filename, writer=writer)

    # Show the plot
    plt.tight_layout()
    plt.show()

    if include_min_distance_plot and include_min_distance_to_obstacles_plot:
        return fig, (ax_traj, ax_goal_dist, ax_min_dist, ax_obstacle_dist)
    elif include_min_distance_plot:
        return fig, (ax_traj, ax_goal_dist, ax_min_dist)
    elif include_min_distance_to_obstacles_plot:
        return fig, (ax_traj, ax_goal_dist, ax_obstacle_dist)
    else:
        return fig, (ax_traj, ax_goal_dist)
