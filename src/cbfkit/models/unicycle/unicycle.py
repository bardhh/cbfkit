import jax.numpy as jnp
import matplotlib

matplotlib.use("macosx")
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, jacfwd
from matplotlib.animation import FuncAnimation
from .barrier_functions import CX, CY, R


#! DYNAMICS
def accel_unicycle_dynamics(stochastic=False):
    """
    Returns a function that computes the unicycle model dynamics.
    """

    @jit
    def dynamics(state):
        """
        Computes the unicycle model dynamics.

        Args:
        state (array-like): The state of the unicycle model, [x, y, v, theta].

        Returns:
        tuple: A tuple containing the function (f) and (g).x
        """
        x, y, v, theta = state
        f = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), 0, 0])
        g = jnp.array([[0, 0], [0, 0], [0, 1], [1, 0]])
        s = jnp.eye(4) * 0.1

        if not stochastic:
            return f, g
        else:
            return f, g, s

    return dynamics


def approx_unicycle_dynamics(l=1.0, stochastic=False):
    """
    Returns a function that represents the approximate unicycle dynamics, which computes
    the drift vector 'f' and control matrix 'g' based on the given state.
    """

    @jit
    def dynamics(state):
        """
        Computes the drift vector 'f' and control matrix 'g' based on the given state.

        :param state: A numpy array representing the current state (x, y, theta, l)
                      where x and y are positions, theta is the orientation angle,
                      and l is the wheelbase of the unicycle.
        :return: A tuple (f, g) where f is the drift vector and g is the control matrix.
        """
        x, y, theta = state
        f = jnp.array([0, 0, 0])

        g = jnp.array(
            [
                [jnp.cos(theta), -l * jnp.sin(theta)],
                [jnp.sin(theta), l * jnp.cos(theta)],
                [0, 1],
            ]
        )
        s = jnp.eye(3) * 0.05

        if not stochastic:
            return f, g
        else:
            return f, g, s

    return dynamics


#! CONTROLLERS
def approx_unicycle_nominal_controller(dynamics, Kp_pos, Kp_theta, desired_state):
    """
    Create a proportional-only controller for the given unicycle dynamics.

    :param dynamics: approximate unicycle dynamics ode
    :param Kp_pos: Position proportional gain.
    :param Kp_theta: Orientation proportional gain.
    :param l: Wheelbase of the unicycle.
    :return: A function that computes control inputs based on the current state and desired state.
    """

    @jit
    def controller(state):
        x, y, theta = state
        x_desired, y_desired, theta_desired = desired_state

        # Compute the error between the current state and the desired state
        error_pos = jnp.subtract(desired_state[:2], state[:2])
        error_theta = theta_desired - theta

        # Compute control inputs based on proportional gains
        control_inputs_pos = jnp.multiply(Kp_pos, error_pos)
        control_input_theta = Kp_theta * error_theta

        # Convert position and orientation control inputs into unicycle control inputs (v, w)
        g = dynamics(state)[1]
        unicycle_control_inputs = jnp.linalg.pinv(g) @ jnp.hstack(
            (control_inputs_pos, control_input_theta)
        )

        return unicycle_control_inputs

    return controller


#! CONTROL BARRIER FUNCTIONS
CX1, CY1, R1 = 1.0, 2.0, 0.5
CX2, CY2, R2 = 2.0, 2.0, 0.75
CX3, CY3, R3 = 0.0, 3.0, 1.0


@jit
def h(x, cx, cy, r):
    return (x[0] - cx) ** 2 + (x[1] - cy) ** 2 - r**2


@jit
def dhdx(x, cx, cy, r):
    return jacfwd(h)(x, cx, cy, r)


def barrier_function(x):
    return h(x, CX, CY, R)


def barrier_jacobian(x):
    return dhdx(x, CX, CY, R)


#! PLOTTING


def plot_trajectory(
    states,
    desired_state,
    desired_state_radius=0.1,
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
    ax.add_patch(
        plt.Circle(
            (CX, CY),
            R,
            color="k",
            fill=True,
            linestyle="-",
            linewidth=1,
        )
    )

    ax.plot(states[:, 0], states[:, 1], label="Trajectory")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid()

    plt.show()

    return fig, ax


def animate(
    states,
    desired_state,
    desired_state_radius=0.1,
    x_lim=(-4, 4),
    y_lim=(-4, 4),
    dt=0.1,
    title="System Behavior",
    save_animation=False,
    animation_filename="system_behavior.gif",
):
    def init():
        trajectory.set_data([], [])
        return (trajectory,)

    def update(frame):
        trajectory.set_data(states[:frame, 0], states[:frame, 1])
        x, y, theta = states[frame]
        return (trajectory,)

    fig, ax = plt.subplots()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    desired_state_radius = 0.1
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
    for x, y, r in zip(CX, CY, R):
        ax.add_patch(
            plt.Circle(
                (x, y),
                r,
                color="k",
                fill=True,
                linestyle="-",
                linewidth=1,
            )
        )

    (trajectory,) = ax.plot([], [], label="Trajectory")

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid()

    ani = FuncAnimation(
        fig, update, frames=len(states), init_func=init, blit=True, interval=dt * 1000
    )

    if save_animation:
        ani.save(animation_filename, writer="imagemagick", fps=15)

    plt.show()

    return fig, ax
