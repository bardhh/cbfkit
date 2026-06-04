from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.lqr import compute_lqr_gain
from cbfkit.utils.matrix_vector_operations import normalize, vee
from cbfkit.utils.user_types import (
    ControllerCallable,
    ControllerCallableReturns,
    ControllerData,
    DynamicsCallable,
)

from ..models.quadrotor_6dof_dynamics import g_accel as g
from ..utils.rotations import rotation_body_frame_to_inertial_frame


def geometric_controller(
    dynamics: DynamicsCallable,
    desired_state: Array,
    dt: float,
    m: float = 4.34,
    jx: float = 0.0820,
    jy: float = 0.0845,
    jz: float = 0.1377,
    kx: float = 8.0,
    kv: float = 8.0,
    kr: float = 8.81,
    ko: float = 2.54,
) -> ControllerCallable:
    """Creates a geometric controller for the 6-DOF quadrotor model based on the following paper:.

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

    Returns
    -------
        controller (Callable): computes control input
    """
    e3 = jnp.array([0.0, 0.0, 1.0])
    j_vec = jnp.array([jx, jy, jz])
    _b1_d = jnp.array([1.0, 0.0, 0.0])

    # The packaged body->inertial matrix is orthogonal but *improper* (det = -1): its
    # third row is the standard ZYX rotation's third row negated, encoding the model's
    # "body z-down, inertial h-up" convention. Geometric SE(3) tracking requires a
    # proper rotation, so we left-multiply by S = diag(1, 1, -1) to flip that row back.
    # The result is a true SO(3) matrix expressed in the z-down ("NED") inertial frame
    # y = S @ p -- exactly the frame the plant's own velocity/gravity terms live in.
    # (Left-multiplying S negates the 3rd row and preserves Rdot = R @ hat([p, q, r]);
    # a right-multiply would silently remap the body rates to [-p, -q, r] and diverge.)
    S = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    pos_d = jnp.matmul(S, desired_state[:3])

    @jit
    def controller(
        t: float,
        x: Array,
        _u_nom: Optional[Array] = None,
        _key: Optional[Array] = None,
        _data: Optional[ControllerData] = None,
    ) -> ControllerCallableReturns:
        """Computes control input.

        Args:
            t (float): time in sec
            x (Array): state vector

        Returns
        -------
            u (Array): computed control inputs
            data (dict): requisite dictionary return
        """
        # Proper SO(3) rotation, body frame -> z-down ("NED") inertial frame.
        rotation = jnp.matmul(S, rotation_body_frame_to_inertial_frame(x))

        # Translational tracking errors in the z-down inertial frame (vel_d = 0).
        e_pos = jnp.matmul(S, x[:3]) - pos_d
        e_vel = jnp.matmul(rotation, x[3:6])

        # Desired thrust direction (Lee et al. 2010, NED form: gravity acts along +e3).
        thrust_vec = -kx * e_pos - kv * e_vel - m * g * e3
        b3_d = -normalize(thrust_vec)
        b2_d = normalize(jnp.cross(b3_d, _b1_d))
        b1_d = normalize(jnp.cross(b2_d, b3_d))
        rot_d = jnp.array([b1_d, b2_d, b3_d]).T

        # Attitude error on SO(3): the vee map is only valid because `rotation` is proper.
        e_rot = 1 / 2 * vee(jnp.matmul(rot_d.T, rotation) - jnp.matmul(rotation.T, rot_d))

        # Body angular velocity [p, q, r] is part of the state; omega_d = 0 for a setpoint.
        omega = x[9:12]
        e_ome = omega

        # Thrust magnitude: project the desired force onto the body-down axis.
        f = -jnp.dot(thrust_vec, jnp.matmul(rotation, e3))

        # Moment inputs (omega_d = omega_d_dot = 0 for setpoint regulation).
        moments = -kr * e_rot - ko * e_ome + jnp.cross(omega, j_vec * omega)

        inputs = jnp.hstack([f, moments])

        return inputs, ControllerData()

    return controller


def lyapunov_control(
    goal: Array,
    dt: float,
    fV: Callable[[Array], float],
    k1: float = 1.0,
    k3: float = 1.0,
    k5: float = 1.0,
):
    """Creates a function to compute the desired accelerations for reaching a goal location based on
    a Lyapunov function.

    Args:
        goal (Array): desired position vector
        dt (float): timestep length
        fV (Callable): Lyapunov function
        k1 (float): positive gain
        k3 (float): positive gain
        k5 (float): positive gain

    Returns
    -------
        Callable: computes the desired position, velocity, and acceleration
    """
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
    """Creates a function to compute the desired accelerations for reaching a goal location based on
    a double integrator model given the time and state vector.

    Args:
        xd (Array): desired position vector
        dt (float): timestep length

    Returns
    -------
        get_desired_pos_vel_acc (Callable): computes the desired position, velocity, and acceleration.
    """
    # Generate A, B, Q, R for LQR
    A = jnp.zeros((6, 6))
    A = A.at[:3, 3:6].set(jnp.eye(3))
    B = jnp.zeros((6, 3))
    B = B.at[3:, :].set(jnp.eye(3))
    Q = jnp.eye(6)
    R = jnp.eye(3)

    # Compute LQR gain
    K = compute_lqr_gain(A, B, Q, R)

    # @jit
    def controller(_t: float, x: Array) -> Tuple[Array, Array, Array]:
        """Computes desired position, velocity, and acceleration based on current and goal states.

        Args:
            t (float): time in sec (unused)
            x (Array): state vector

        Returns
        -------
            xd (Array): goal position vector
            vd (Array): goal velocity vector
            ad (Array): goal acceleration vector
        """
        pos_vel = jnp.hstack([x[:3], jnp.matmul(rotation_body_frame_to_inertial_frame(x), x[3:6])])
        ad = -jnp.matmul(K, pos_vel - jnp.hstack([xd, jnp.zeros((3,))]))
        vd = pos_vel[3:6] + ad * dt

        return xd, vd, ad

    return controller
