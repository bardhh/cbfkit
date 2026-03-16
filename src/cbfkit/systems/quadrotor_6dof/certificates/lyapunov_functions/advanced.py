"""Advanced Lyapunov functions: composite, position-velocity, and geometric convergence."""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit

from cbfkit.utils.lqr import compute_lqr_gain
from cbfkit.utils.matrix_vector_operations import normalize, vee
from cbfkit.utils.user_types import CertificateCollection

from ...models.quadrotor_6dof_dynamics import rotation_body_to_inertial
from ...utils.rotations import rotation_body_frame_to_inertial_frame
from .basic import N, G, V_att, V_pos, V_vel


###############################################################################
#! Composite Convergence
def V_com(
    state: Array,
    control: Callable[[float, Array], Tuple[Array, Array, Array]],
    goal: Array,
    lam: float,
) -> Array:
    """Composite convergence function (takes V_pos, V_att, V_vel into account)."""
    arr = jnp.array([V_pos(state, goal), V_att(state, lam), V_vel(state, control)])

    return jnp.sum(arr)


@jit
def dV_com_dx(
    z: Array, control: Callable[[float, Array], Tuple[Array, Array, Array]], goal: Array, lam: float
) -> Array:
    """Jacobian for composite convergence function."""
    return jacfwd(V_com)(z, control, goal, lam)


@jit
def dV2_com_dx2(
    z: Array, control: Callable[[float, Array], Tuple[Array, Array, Array]], goal: Array, lam: float
) -> Array:
    """Hessian for composite convergence function."""
    return jacfwd(jacrev(V_com))(z, control, goal, lam)


def composite(goal: Array, lam: float = 0.25) -> CertificateCollection:
    """Generates Lyapunov function for composite convergence.

    Args:
        goal (Array): goal position vector [xd, yd, zd]
        lam (float): length of lever arm

    Returns
    -------
        CertificateCollection: certificate functions
    """
    control = double_integrator_control(goal, 0.01)

    def v_func(t, x):
        return V_com(jnp.hstack([x, t]), control, goal, lam)  # type: ignore[return-value]

    def j_func(t, x):
        return dV_com_dx(jnp.hstack([x, t]), control, goal, lam)[:N]  # type: ignore[return-value]

    def h_func(t, x):
        return dV2_com_dx2(jnp.hstack([x, t]), control, goal, lam)[:N, :N]  # type: ignore[return-value]

    def p_func(t, x):
        return dV_com_dx(jnp.hstack([x, t]), control, goal, lam)[-1]  # type: ignore[return-value]

    return CertificateCollection(
        [v_func],
        [j_func],
        [h_func],
        [p_func],
        [],
    )


###############################################################################
#! Position-Velocity Convergence
@jit
def V_pv(state: Array, goal: Array, k1: float = 1.0, k3: float = 1.0, k5: float = 1.0) -> Array:
    """Position/velocity goal function (drive quadrotor to goal location)."""
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
    """Jacobian for position-velocity goal function."""
    return jacfwd(V_pv)(z, goal, k1, k3, k5)


@jit
def dV2_pv_dx2(z: Array, goal: Array, k1: float = 1.0, k3: float = 1.0, k5: float = 1.0) -> Array:
    """Hessian for position-velocity goal function."""
    return jacfwd(jacrev(V_pv))(z, goal, k1, k3, k5)


def position_velocity(goal: Array, k1: float, k3: float, k5: float) -> CertificateCollection:
    """Generates Lyapunov function for position and velocity convergence.

    Args:
        goal (Array): goal position vector [xd, yd, zd]
        k1, k3, k5 (float): positive gains

    Returns
    -------
        CertificateCollection: certificate functions
    """

    def v_func(t, x):
        return V_pv(jnp.hstack([x, t]), goal, k1, k3, k5)  # type: ignore[return-value]

    def j_func(t, x):
        return dV_pv_dx(jnp.hstack([x, t]), goal, k1, k3, k5)[:N]  # type: ignore[return-value]

    def h_func(t, x):
        return dV2_pv_dx2(jnp.hstack([x, t]), goal, k1, k3, k5)[:N, :N]  # type: ignore[return-value]

    def p_func(t, x):
        return dV_pv_dx(jnp.hstack([x, t]), goal, k1, k3, k5)[-1]  # type: ignore[return-value]

    return CertificateCollection(
        [v_func],
        [j_func],
        [h_func],
        [p_func],
        [],
    )


###############################################################################
#! Geometric Error Convergence
def V_geo(
    state: Array,
    goal: Array,
    k_lqr: Array,
    m: float,
    kx: float,
    kv: float,
) -> Array:
    """Geometric error goal function (drive quadrotor to goal location)."""
    x, y, z, u, v, w, phi, theta, psi, p, q, r, _ = state

    body_to_inertial_rotation = rotation_body_frame_to_inertial_frame(state)

    pos_inertial = jnp.array([x, y, z])
    vel_inertial = jnp.matmul(body_to_inertial_rotation, jnp.array([u, v, w]))

    phi_dot = p + q * jnp.sin(phi) * jnp.tan(theta) + r * jnp.cos(phi) * jnp.tan(theta)
    theta_dot = q * jnp.cos(phi) - r * jnp.sin(phi)
    psi_dot = q * jnp.sin(phi) / jnp.cos(theta) + r * jnp.cos(phi) / jnp.cos(theta)

    pos_d = goal
    vel_d = -jnp.matmul(k_lqr, pos_inertial - pos_d)
    acc_d0 = 0.25 * (vel_d - vel_inertial)
    acc_d = (
        jnp.sign(acc_d0) * jnp.multiply(acc_d0, acc_d0) ** 0.5
        + jnp.sign(acc_d0) * jnp.multiply(acc_d0, acc_d0) ** 1.5
    )

    e_pos = pos_inertial - pos_d
    e_vel = vel_inertial - vel_d

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

    wx_b = phi_dot * jnp.sin(theta) * jnp.sin(psi) + theta_dot * jnp.cos(psi)
    wy_b = phi_dot * jnp.sin(theta) * jnp.cos(psi) - theta_dot * jnp.sin(psi)
    wz_b = phi_dot * jnp.cos(theta) + psi_dot
    omega = jnp.array([wx_b, wy_b, wz_b])

    omega_d = jnp.zeros((3,))
    e_ome = omega - jnp.matmul(body_to_inertial_rotation.T, jnp.matmul(rot_d, omega_d))

    V = 0.5 * (
        10 * jnp.dot(e_pos, e_pos)
        + jnp.dot(e_vel, e_vel)
        + jnp.dot(e_rot, e_rot)
        + jnp.dot(e_ome, e_ome)
    )

    -jnp.dot(
        -kx * e_pos - kv * e_vel + m * G * e3_ + m * acc_d,
        jnp.matmul(body_to_inertial_rotation, e3_),
    )

    return V


@jit
def dV_geo_dx(
    z: Array, goal: Array, k_lqr: Array, m: float, kx: float = 1.0, kv: float = 1.0
) -> Array:
    """Jacobian for geometric error goal function."""
    return jacfwd(V_geo)(z, goal, k_lqr, m, kx, kv)


@jit
def dV2_geo_dx2(
    z: Array, goal: Array, k_lqr: Array, m: float, kx: float = 1.0, kv: float = 1.0
) -> Array:
    """Hessian for geometric error goal function."""
    return jacfwd(jacrev(V_geo))(z, goal, k_lqr, m, kx, kv)


def geometric(xd: Array, kx: float, kv: float, m: float, g: float) -> CertificateCollection:
    """Generates Geometric Lyapunov function.

    Args:
        xd (Array): desired state
        kx, kv, m, g: parameters

    Returns
    -------
        CertificateCollection: certificate functions
    """

    def v_func(t, x):
        return V_geo(jnp.hstack([x, t]), xd, jnp.zeros((3, 3)), m, kx, kv)  # type: ignore[return-value]

    def j_func(t, x):
        return dV_geo_dx(jnp.hstack([x, t]), xd, jnp.zeros((3, 3)), m, kx, kv)[:N]  # type: ignore[return-value]

    def h_func(t, x):
        return dV2_geo_dx2(jnp.hstack([x, t]), xd, jnp.zeros((3, 3)), m, kx, kv)[:N, :N]  # type: ignore[return-value]

    def p_func(t, x):
        return dV_geo_dx(jnp.hstack([x, t]), xd, jnp.zeros((3, 3)), m, kx, kv)[-1]  # type: ignore[return-value]

    return CertificateCollection(
        [v_func],
        [j_func],
        [h_func],
        [p_func],
        [],
    )


###############################################################################
#! Helper Functions
def double_integrator_control(
    xd: Array, dt: float
) -> Callable[[float, Array], Tuple[Array, Array, Array]]:
    """Creates a function to compute desired accelerations for reaching a goal location
    based on a double integrator model given the time and state vector.

    Args:
        xd (Array): desired position vector
        dt (float): timestep length

    Returns
    -------
        Callable: computes the desired position, velocity, and acceleration.
    """
    A = jnp.zeros((6, 6))
    A = A.at[:3, 3:6].set(jnp.eye(3))
    B = jnp.zeros((6, 3))
    B = B.at[3:, :].set(jnp.eye(3))
    Q = jnp.eye(6)
    R = jnp.eye(3)

    K = compute_lqr_gain(A, B, Q, R)

    @jit
    def lqr_control(_t: float, x: Array) -> Tuple[Array, Array, Array]:
        phi, theta, psi = x[6], x[7], x[8]
        pos_vel = jnp.hstack(
            [x[:3], jnp.matmul(rotation_body_to_inertial(phi, theta, psi), x[3:6])]
        )
        ad = -jnp.matmul(K, pos_vel - jnp.hstack([xd, jnp.zeros((3,))]))
        vd = pos_vel[3:6] + ad * dt

        return jnp.zeros((3,)), vd, ad

    return lqr_control
