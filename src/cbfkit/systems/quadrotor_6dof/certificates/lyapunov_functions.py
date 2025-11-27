import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array
from typing import Tuple, Callable
from control import lqr
from cbfkit.utils.matrix_vector_operations import normalize, vee
from ..utils.rotations import rotation_body_frame_to_inertial_frame
from ..models.quadrotor_6dof_dynamics import rotation_body_to_inertial
from cbfkit.utils.user_types import (
    CertificateCallable,
    CertificateJacobianCallable,
    CertificateHessianCallable,
    CertificatePartialCallable,
    CertificateTuple,
)

# constants
N = 12  # number of states
G = 9.81  # accel due to gravity (m/s^2)


###############################################################################
#! Position Convergence
@jit
def V_pos(state: Array, goal: Array) -> Array:
    """Position goal function (drive quadrotor to goal location).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    x, y, z, _, _, _, _, _, _, _, _, _, _ = state
    V = 0.5 * ((x - goal[0]) ** 2 + (y - goal[1]) ** 2 + (z - goal[2]) ** 2)

    return V


@jit
def dV_pos_dx(z: Array, goal: Array) -> Array:
    """Jacobian for position goal function (drive quadrotor to goal location).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(V_pos)(z, goal)


@jit
def dV2_pos_dx2(z: Array, goal: Array) -> Array:
    """Hessian for position goal function (drive quadrotor to goal location).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(V_pos))(z, goal)


def position(goal: Array) -> CertificateTuple:
    v_func: CertificateCallable = lambda t, x: V_pos(jnp.hstack([x, t]), goal)  # type: ignore[return-value]
    j_func: CertificateJacobianCallable = lambda t, x: dV_pos_dx(jnp.hstack([x, t]), goal)[:N]  # type: ignore[return-value]
    h_func: CertificateHessianCallable = lambda t, x: dV2_pos_dx2(jnp.hstack([x, t]), goal)[:N, :N]  # type: ignore[return-value]
    p_func: CertificatePartialCallable = lambda t, x: dV_pos_dx(jnp.hstack([x, t]), goal)[-1]  # type: ignore[return-value]

    return (
        v_func,
        j_func,
        h_func,
        p_func,
    )


###############################################################################
#! Attitude Convergence
@jit
def V_att(state: Array, l: float) -> Array:
    """Attitude convergence function (quadrotor is in hover-mode at goal location).

    Arguments:
        z (Array): concatenated time and state vector
        l (float): length of lever arm

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    x, y, z, _, _, _, phi, theta, psi, _, _, _, _ = state
    Rr = rotation_body_to_inertial(phi, theta, psi)
    V = 0.5 * jnp.linalg.norm(jnp.array([x, y, z]) + l * jnp.matmul(Rr, jnp.array([1, 1, 1])))

    return V


@jit
def dV_att_dx(z: Array, l: float) -> Array:
    """Jacobian for attitude convergence function (quadrotor is in hover-mode at goal location).

    Arguments:
        z (Array): concatenated time and state vector
        l (float): length of lever arm

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(V_att)(z, l)


@jit
def dV2_att_dx2(z: Array, l: float) -> Array:
    """Hessian for attitude convergence function (quadrotor is in hover-mode at goal location).

    Arguments:
        z (Array): concatenated time and state vector
        l (float): length of lever arm

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(V_att))(z, l)


def attitude(l: float = 0.25) -> CertificateTuple:
    v_func: CertificateCallable = lambda t, x: V_att(jnp.hstack([x, t]), l)  # type: ignore[return-value]
    j_func: CertificateJacobianCallable = lambda t, x: dV_att_dx(jnp.hstack([x, t]), l)[:N]  # type: ignore[return-value]
    h_func: CertificateHessianCallable = lambda t, x: dV2_att_dx2(jnp.hstack([x, t]), l)[:N, :N]  # type: ignore[return-value]
    p_func: CertificatePartialCallable = lambda t, x: dV_att_dx(jnp.hstack([x, t]), l)[-1]  # type: ignore[return-value]

    return (
        v_func,
        j_func,
        h_func,
        p_func,
    )


###############################################################################
#! Velocity Convergence
@jit
def V_vel(state: Array, control: Callable[[float, Array], Tuple[Array]]) -> Array:
    """Velocity convergence function (quadrotor has zero translational and rotational velocity at goal).

    Arguments:
        z (Array): concatenated time and state vector

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    _, _, _, u, v, w, _, _, _, p, q, r, _ = state
    Rr = rotation_body_to_inertial(state)
    _, vd, ad = control(state)
    R13_c = -ad[0] / 9.81
    R23_c = -ad[1] / 9.81
    R13dot_c = R13_c - Rr[0, 2]
    R23dot_c = R23_c - Rr[1, 2]
    u0, v0, w0 = jnp.matmul(Rr.T, vd)
    p0 = -R13dot_c * Rr[0, 1] - R23dot_c * Rr[1, 1]
    q0 = R13dot_c * Rr[0, 0] + R23dot_c * Rr[1, 2]
    r0 = 0.0

    V = 0.5 * (
        (u - u0) ** 2
        + (v - v0) ** 2
        + (w - w0) ** 2
        + (p - p0) ** 2
        + (q - q0) ** 2
        + (r - r0) ** 2
    )

    return V


@jit
def dV_vel_dx(z: Array) -> Array:
    """Jacobian for velocity convergence function (quadrotor has zero translational and rotational velocity at goal).

    Arguments:
        z (Array): concatenated time and state vector

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(V_vel)(z)


@jit
def dV2_vel_dx2(z: Array) -> Array:
    """Hessian for velocity convergence function (quadrotor has zero translational and rotational velocity at goal).

    Arguments:
        z (Array): concatenated time and state vector

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(V_vel))(z)


def velocity(goal: Array) -> CertificateTuple:
    control = double_integrator_control(goal, 0.01)
    v_func: CertificateCallable = lambda t, x: V_vel(jnp.hstack([x, t]), control)  # type: ignore[return-value]
    j_func: CertificateJacobianCallable = lambda t, x: dV_vel_dx(jnp.hstack([x, t]), control)[:N]  # type: ignore[return-value]
    h_func: CertificateHessianCallable = lambda t, x: dV2_vel_dx2(jnp.hstack([x, t]), control)[:N, :N]  # type: ignore[return-value]
    p_func: CertificatePartialCallable = lambda t, x: dV_vel_dx(jnp.hstack([x, t]), control)[-1]  # type: ignore[return-value]

    return (
        v_func,
        j_func,
        h_func,
        p_func,
    )


###############################################################################
#! Composite Convergence
# @jit
def V_com(
    state: Array, control: Callable[[float, Array], Tuple[Array]], goal: Array, l: float
) -> Array:
    """Composite convergence function (takes V_pos, V_att, V_vel into account).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]
        l (float): length of lever arm

    Returns:
        ret (float): value of composite function evaluated at time and state
    """
    arr = jnp.array([V_pos(state, goal), V_att(state, l), V_vel(state, control)])

    return jnp.sum(arr)


@jit
def dV_com_dx(
    z: Array, control: Callable[[float, Array], Tuple[Array]], goal: Array, l: float
) -> Array:
    """Jacobian for composite convergence function (takes V_pos, V_att, V_vel into account).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]
        l (float): length of lever arm

    Returns:
        ret (float): value of composite function evaluated at time and state
    """
    return jacfwd(V_com)(z, control, goal, l)


@jit
def dV2_vel_dx2(
    z: Array, control: Callable[[float, Array], Tuple[Array]], goal: Array, l: float
) -> Array:
    """Hessian for composite convergence function (takes V_pos, V_att, V_vel into account).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]
        l (float): length of lever arm

    Returns:
        ret (float): value of composite function evaluated at time and state
    """
    return jacfwd(jacrev(V_com))(z, control, goal, l)


def composite(goal: Array, l: float = 0.25) -> CertificateTuple:
    control = double_integrator_control(goal, 0.01)
    v_func: CertificateCallable = lambda t, x: V_com(jnp.hstack([x, t]), control, goal, l)  # type: ignore[return-value]
    j_func: CertificateJacobianCallable = lambda t, x: dV_com_dx(jnp.hstack([x, t]), control, goal, l)[:N]  # type: ignore[return-value]
    h_func: CertificateHessianCallable = lambda t, x: dV2_com_dx2(jnp.hstack([x, t]), control, goal, l)[:N, :N]  # type: ignore[return-value]
    p_func: CertificatePartialCallable = lambda t, x: dV_com_dx(jnp.hstack([x, t]), control, goal, l)[-1]  # type: ignore[return-value]

    return (
        [v_func],
        [j_func],
        [h_func],
        [p_func],
    )


###############################################################################
#! Position Convergence
@jit
def V_pv(state: Array, goal: Array, k1: float = 1.0, k3: float = 1.0, k5: float = 1.0) -> Array:
    """Position/velocity goal function (drive quadrotor to goal location).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]
        k1 (float): positive gain
        k3 (float): positive gain
        k5 (float): positive gain

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    x, y, z, u, v, w, phi, theta, psi, _, _, _, _ = state
    x_dot = (
        u * (jnp.cos(theta) * jnp.cos(psi))
        + v * (jnp.sin(phi) * jnp.sin(theta) * jnp.cos(psi) - jnp.cos(phi) * jnp.sin(psi))
        + w * (jnp.cos(phi) * jnp.sin(theta) * jnp.cos(psi) + jnp.sin(phi) * jnp.sin(psi))
    )
    y_dot = (
        u * (jnp.cos(theta) * jnp.sin(psi))
        + v * (jnp.sin(phi) * jnp.sin(theta) * jnp.sin(psi) + jnp.cos(phi) * jnp.cos(psi))
        + w * (jnp.cos(phi) * jnp.sin(theta) * jnp.sin(psi) - jnp.sin(phi) * jnp.cos(psi))
    )
    z_dot = (
        u * jnp.sin(theta) - v * jnp.sin(phi) * jnp.cos(theta) - w * jnp.cos(phi) * jnp.cos(theta)
    )

    V = 0.5 * (
        (x - goal[0]) ** 2
        + (x_dot + k1 * (x - goal[0])) ** 2
        + (y - goal[1]) ** 2
        + (y_dot + k3 * (y - goal[1])) ** 2
        + (z - goal[2]) ** 2
        + (z_dot + k5 * (z - goal[2])) ** 2
    )

    return V


@jit
def dV_pv_dx(z: Array, goal: Array, k1: float = 1.0, k3: float = 1.0, k5: float = 1.0) -> Array:
    """Jacobian for position goal function (drive quadrotor to goal location).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(V_pv)(z, goal, k1, k3, k5)


@jit
def dV2_pv_dx2(z: Array, goal: Array, k1: float = 1.0, k3: float = 1.0, k5: float = 1.0) -> Array:
    """Hessian for position goal function (drive quadrotor to goal location).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(V_pv))(z, goal, k1, k3, k5)


def position_velocity(goal: Array, k1: float, k3: float, k5: float) -> CertificateTuple:
    v_func: CertificateCallable = lambda t, x: V_pv(jnp.hstack([x, t]), goal, k1, k3, k5)  # type: ignore[return-value]
    j_func: CertificateJacobianCallable = lambda t, x: dV_pv_dx(jnp.hstack([x, t]), goal, k1, k3, k5)[:N]  # type: ignore[return-value]
    h_func: CertificateHessianCallable = lambda t, x: dV2_pv_dx2(jnp.hstack([x, t]), goal, k1, k3, k5)[:N, :N]  # type: ignore[return-value]
    p_func: CertificatePartialCallable = lambda t, x: dV_pv_dx(jnp.hstack([x, t]), goal, k1, k3, k5)[-1]  # type: ignore[return-value]

    return (
        v_func,
        j_func,
        h_func,
        p_func,
    )


###############################################################################
#! Geometric Error Convergence
# @jit
def V_geo(
    state: Array,
    goal: Array,
    k_lqr: Array,
    m: float,
    kx: float,
    kv: float,
) -> Array:
    """Geometric error goal function (drive quadrotor to goal location).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]
        k_lqr (Array): kqr gain for double integrator control
        m (float): quadrotor mass in kg
        kx (float): position error gain
        kv (float): velocity error gain

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    x, y, z, u, v, w, phi, theta, psi, p, q, r, _ = state

    # Get rotation matrix
    body_to_inertial_rotation = rotation_body_frame_to_inertial_frame(state)

    # Get inertial position/velocity
    pos_inertial = jnp.array([x, y, z])
    vel_inertial = jnp.matmul(body_to_inertial_rotation, jnp.array([u, v, w]))

    # State derivatives (copied from model, JIT does not take well to callable functions)
    phi_dot = p + q * jnp.sin(phi) * jnp.tan(theta) + r * jnp.cos(phi) * jnp.tan(theta)
    theta_dot = q * jnp.cos(phi) - r * jnp.sin(phi)
    psi_dot = q * jnp.sin(phi) / jnp.cos(theta) + r * jnp.cos(phi) / jnp.cos(theta)

    # Compute desired position, velocity, acceleration
    pos_d = goal
    vel_d = -jnp.matmul(k_lqr, pos_inertial - pos_d)
    acc_d0 = 0.25 * (vel_d - vel_inertial)
    acc_d = (
        jnp.sign(acc_d0) * jnp.multiply(acc_d0, acc_d0) ** 0.5
        + jnp.sign(acc_d0) * jnp.multiply(acc_d0, acc_d0) ** 1.5
    )

    # Define tracking errors
    e_pos = pos_inertial - pos_d
    e_vel = vel_inertial - vel_d

    # Compute desired attitude and attitude tracking error
    e3_ = jnp.array([0.0, 0.0, 1.0])
    b1_d_ = jnp.array([1.0, 0.0, 0.0])
    b3_d = -normalize(-kx * e_pos - kv * e_vel - m * G * e3_ + m * acc_d)
    b2_d = normalize(jnp.cross(b3_d, b1_d_))
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
    e_ome = omega - jnp.matmul(body_to_inertial_rotation.T, jnp.matmul(rot_d, omega_d))

    V = 0.5 * (
        10 * jnp.dot(e_pos, e_pos)
        + jnp.dot(e_vel, e_vel)
        + jnp.dot(e_rot, e_rot)
        + jnp.dot(e_ome, e_ome)
    )

    f = -jnp.dot(
        -kx * e_pos - kv * e_vel + m * G * e3_ + m * acc_d,
        jnp.matmul(body_to_inertial_rotation, e3_),
    )

    return V


@jit
def dV_geo_dx(
    z: Array, goal: Array, k_lqr: Array, m: float, kx: float = 1.0, kv: float = 1.0
) -> Array:
    """Jacobian for geometric error goal function (drive quadrotor to goal location).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]
        k_lqr (Array): kqr gain for double integrator control
        m (float): quadrotor mass in kg
        kx (float): position error gain
        kv (float): velocity error gain

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(V_geo)(z, goal, k_lqr, m, kx, kv)


@jit
def dV2_geo_dx2(
    z: Array, goal: Array, k_lqr: Array, m: float, kx: float = 1.0, kv: float = 1.0
) -> Array:
    """Hessian for geometric error goal function (drive quadrotor to goal location).

    Arguments:
        z (Array): concatenated time and state vector
        goal (Array): goal position vector [xd, yd, zd]
        k_lqr (Array): kqr gain for double integrator control
        m (float): quadrotor mass in kg
        kx (float): position error gain
        kv (float): velocity error gain

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(V_geo))(z, goal, k_lqr, m, kx, kv)


def geometric(goal: Array, k_lqr: Array, m: float, kx: float, kv: float) -> CertificateTuple:
    v_func: CertificateCallable = lambda t, x: V_geo(jnp.hstack([x, t]), goal, k_lqr, m, kx, kv)  # type: ignore[return-value]
    j_func: CertificateJacobianCallable = lambda t, x: dV_geo_dx(jnp.hstack([x, t]), goal, k_lqr, m, kx, kv)[:N]  # type: ignore[return-value]
    h_func: CertificateHessianCallable = lambda t, x: dV2_geo_dx2(jnp.hstack([x, t]), goal, k_lqr, m, kx, kv)[:N, :N]  # type: ignore[return-value]
    p_func: CertificatePartialCallable = lambda t, x: dV_geo_dx(jnp.hstack([x, t]), goal, k_lqr, m, kx, kv)[-1]  # type: ignore[return-value]

    return (
        [v_func],
        [j_func],
        [h_func],
        [p_func],
    )


# ###############################################################################
# #! Helper Functions
# def nominal_translational_velocity(
#     xd: Array, dt: float
# ) -> Callable[[float, Array], Tuple[Array, Array, Array]]:
#     """
#     Creates a function to compute the desired accelerations for reaching a goal
#     location based on a double integrator model given the time and state vector.

#     Args:
#         xd (Array): desired position vector
#         dt (float): timestep length

#     Returns:
#         get_desired_pos_vel_acc (Callable): computes the desired position, velocity, and acceleration

#     """
#     # Generate A, B, Q, R for LQR
#     A = jnp.zeros((6, 6))
#     A = A.at[:3, 3:6].set(jnp.eye(3))
#     B = jnp.zeros((6, 3))
#     B = B.at[3:, :].set(jnp.eye(3))
#     Q = jnp.eye(6)
#     R = jnp.eye(3)

#     # Compute LQR gain
#     K, _, _ = lqr(A, B, Q, R)

#     @jit
#     def lqr_control(_t: float, x: Array) -> Tuple[Array, Array, Array]:
#         """
#         Computes desired position, velocity, and acceleration based on current and goal
#         states.

#         Args:
#             t (float): time in sec (unused)
#             x (Array): state vector

#         Returns:
#             xd (Array): goal position vector
#             vd (Array): goal velocity vector
#             ad (Array): goal acceleration vector

#         """
#         pos_vel = jnp.hstack([x[:3], jnp.matmul(rotation_body_to_inertial(x), x[3:6])])
#         ad = -jnp.matmul(K, pos_vel - jnp.hstack([xd, jnp.zeros((3,))]))
#         vd = pos_vel[3:6] + ad * dt

#         return xd, vd, ad

#     return lqr_control
