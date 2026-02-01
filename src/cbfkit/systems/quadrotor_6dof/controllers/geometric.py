import jax.numpy as jnp
from typing import Callable, Tuple
from jax import jit, Array
from control import lqr
from cbfkit.utils.matrix_vector_operations import normalize, hat, vee
from cbfkit.utils.user_types import DynamicsCallable, ControllerCallable, ControllerCallableReturns
from ..utils.rotations import rotation_body_frame_to_inertial_frame
from ..models.quadrotor_6dof_dynamics import g_accel as g
from ..certificate_functions.lyapunov_functions import V_pv as V


def geometric_controller(
    dynamics: DynamicsCallable,
    desired_state: Array,
    dt: float,
    # m: float = 0.5,
    # jx: float = 0.25,
    # jy: float = 0.25,
    # jz: float = 0.1,
    # kx: float = 1.0,
    # kv: float = 2.05,
    # kr: float = 0.35,
    # ko: float = 0.15,
    m: float = 4.34,
    jx: float = 0.0820,
    jy: float = 0.0845,
    jz: float = 0.1377,
    kx: float = 16.0,
    kv: float = 5.6,
    kr: float = 8.81,
    ko: float = 2.54,
) -> ControllerCallable:
    """
    Creates a geometric controller for the 6-DOF quadrotor model based on the following paper:

    T. Lee, M. Leok and N. H. McClamroch,
        "Geometric tracking control of a quadrotor UAV on SE(3),"
        49th IEEE Conference on Decision and Control (CDC), 2010,
        pp. 5420-5425, doi: 10.1109/CDC.2010.5717652.

    Args:
        dynamics (Callable): computes the dynamics based on the state
        desired_state (Array): goal location
        dt (float): timestep length in sec
        m (float): mass in kg
        jx (float): x moment of inertia
        jy (float): y moment of inertia
        jz (float): z moment of inertia
        kx (float): position error gain
        kv (float): velocity error gain
        kr (float): attitude error gain
        ko (float): omega error gain

    Returns:
        controller (Callable): computes control input
    """
    e3 = jnp.array([0.0, 0.0, 1.0])
    j_vec = jnp.array([jx, jy, jz])
    _b1_d = jnp.array([1.0, 0.0, 0.0])

    # # Flatness-based control -- LQR
    # get_desired_pos_vel_acc = lqr_control(desired_state, dt)

    # Flatness-based control -- FxTS
    tg = 10.0
    c1, e1, e2 = 0.5, 0.5, 1.5
    c2 = 1 / ((e2 - 1) * (tg - 1 / (c1 * (1 - e1))))
    fV = lambda x: -c1 * V(x, desired_state) ** e1 - c2 * V(x, desired_state) ** e2
    get_desired_pos_vel_acc = lyapunov_control(desired_state, dt, fV)

    @jit
    def controller(t: float, x: Array) -> ControllerCallableReturns:
        """
        Computes control input.

        Args:
            t (float): time in sec
            x (Array): state vector

        Returns:
            u (Array): computed control inputs
            data (dict): requisite dictionary return

        """
        nonlocal _b1_d, e3, j_vec

        _, _, _, _, _, _, _, theta, psi, _, _, _ = x
        _, _, _, _, _, _, phi_dot, theta_dot, psi_dot, _, _, _ = dynamics(x)[0]

        # Get rotation matrix
        body_to_inertial_rotation = rotation_body_frame_to_inertial_frame(x)

        # Compute desired position, velocity, acceleration
        pos_d, vel_d, acc_d = get_desired_pos_vel_acc(t, x)
        vel_d = jnp.zeros((3,))
        acc_d = jnp.zeros((3,))

        # Define tracking errors
        e_pos = x[:3] - pos_d
        e_vel = jnp.matmul(body_to_inertial_rotation, x[3:6]) - vel_d

        # Compute desired attitude and attitude tracking error
        b3_d = -normalize(-kx * e_pos - kv * e_vel + m * g * e3 + m * acc_d)
        b2_d = normalize(jnp.cross(b3_d, _b1_d))
        b1_d = normalize(jnp.cross(b2_d, b3_d))
        rot_d = jnp.array([b1_d, b2_d, b3_d]).T
        e_rot = (
            1
            / 2
            * vee(
                jnp.matmul(rot_d.T, body_to_inertial_rotation)
                - jnp.matmul(body_to_inertial_rotation.T, rot_d)
            )
        )

        # Compute angular velocity tracking error
        wx_b = phi_dot * jnp.sin(theta) * jnp.sin(psi) + theta_dot * jnp.cos(psi)
        wy_b = phi_dot * jnp.sin(theta) * jnp.cos(psi) - theta_dot * jnp.sin(psi)
        wz_b = phi_dot * jnp.cos(theta) + psi_dot
        omega = jnp.array([wx_b, wy_b, wz_b])

        # Compute rotation tracking error
        omega_d = jnp.zeros((3,))
        omega_d_dot = jnp.zeros((3,))
        e_ome = omega - jnp.matmul(body_to_inertial_rotation.T, jnp.matmul(rot_d, omega_d))

        # Compute force input
        f = -jnp.dot(
            -kx * e_pos - kv * e_vel + m * g * e3 + m * acc_d,
            jnp.matmul(body_to_inertial_rotation, e3),
        )

        # Compute moment inputs
        #! need to double check this
        moments = (
            -kr * e_rot
            - ko * e_ome
            + jnp.cross(omega, j_vec * omega)
            - j_vec
            * (
                jnp.matmul(
                    jnp.matmul(hat(omega), body_to_inertial_rotation.T), jnp.matmul(rot_d, omega_d)
                )
                - jnp.matmul(body_to_inertial_rotation.T, jnp.matmul(rot_d, omega_d_dot))
            )
        )

        inputs = jnp.hstack([f, moments])

        return inputs, {}

    return controller


def lyapunov_control(
    goal: Array,
    dt: float,
    fV: Callable[[Array], float],
    k1: float = 1.0,
    k3: float = 1.0,
    k5: float = 1.0,
):
    exp = 4
    x_d, y_d, z_d = goal
    x_dot_d, y_dot_d, z_dot_d = 0.0, 0.0, 0.0
    x_2dot_d, y_2dot_d, z_2dot_d = 0.0, 0.0, 0.0

    @jit
    def controller(t: float, state: Array):
        x, y, z, u, v, w, _, _, _, _, _, _ = state
        augmented_state = jnp.hstack([state, t])
        lyapunov = fV(augmented_state)

        vel = jnp.matmul(rotation_body_frame_to_inertial_frame(state), jnp.array([u, v, w]))
        x_dot, y_dot, z_dot = vel

        # Compute accelerations
        x_2dot = (
            x_2dot_d
            - k1 * (x_dot - x_dot_d)
            + (
                ((x - x_d) ** exp * lyapunov)
                / ((x - x_d) ** exp + (y - y_d) ** exp + (z - z_d) ** exp)
                - (x - x_d) * (x_dot - x_dot_d)
            )
            / (x_dot - x_dot_d + k1 * (x - x_d))
        )
        y_2dot = (
            y_2dot_d
            - k3 * (y_dot - y_dot_d)
            + (
                ((y - y_d) ** exp * lyapunov)
                / ((x - x_d) ** exp + (y - y_d) ** exp + (z - z_d) ** exp)
                - (y - y_d) * (y_dot - y_dot_d)
            )
            / (y_dot - y_dot_d + k3 * (y - y_d))
        )
        z_2dot = (
            z_2dot_d
            - k5 * (z_dot - z_dot_d)
            + (
                ((z - z_d) ** exp * lyapunov)
                / ((x - x_d) ** exp + (y - y_d) ** exp + (z - z_d) ** exp)
                - (z - z_d) * (z_dot - z_dot_d)
            )
            / (z_dot - z_dot_d + k5 * (z - z_d))
        )

        pd = jnp.array([x_d, y_d, z_d])
        vd = jnp.array([x_dot + x_2dot * dt, y_dot + y_2dot * dt, z_dot + z_2dot * dt])
        ad = jnp.array([x_2dot, y_2dot, z_2dot])
        return pd, vd, ad

    return controller


def lqr_control(xd: Array, dt: float) -> Callable[[float, Array], Tuple[Array, Array, Array]]:
    """
    Creates a function to compute the desired accelerations for reaching a goal
    location based on a double integrator model given the time and state vector.

    Args:
        xd (Array): desired position vector
        dt (float): timestep length

    Returns:
        get_desired_pos_vel_acc (Callable): computes the desired position, velocity, and acceleration

    """
    # Generate A, B, Q, R for LQR
    A = jnp.zeros((6, 6))
    A = A.at[:3, 3:6].set(jnp.eye(3))
    B = jnp.zeros((6, 3))
    B = B.at[3:, :].set(jnp.eye(3))
    Q = jnp.eye(6)
    R = jnp.eye(3)

    # Compute LQR gain
    K, _, _ = lqr(A, B, Q, R)

    # @jit
    def controller(_t: float, x: Array) -> Tuple[Array, Array, Array]:
        """
        Computes desired position, velocity, and acceleration based on current and goal
        states.

        Args:
            t (float): time in sec (unused)
            x (Array): state vector

        Returns:
            xd (Array): goal position vector
            vd (Array): goal velocity vector
            ad (Array): goal acceleration vector

        """
        pos_vel = jnp.hstack([x[:3], jnp.matmul(rotation_body_frame_to_inertial_frame(x), x[3:6])])
        ad = -jnp.matmul(K, pos_vel - jnp.hstack([xd, jnp.zeros((3,))]))
        vd = pos_vel[3:6] + ad * dt

        return xd, vd, ad

    return controller
