"""Basic Lyapunov functions: position, attitude, and velocity convergence."""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit

from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CertificateCollection,
)

from ...models.quadrotor_6dof_dynamics import rotation_body_to_inertial

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

    Returns
    -------
        ret (float): value of goal function evaluated at time and state
    """
    x, y, z, _, _, _, _, _, _, _, _, _, _ = state
    V = 0.5 * ((x - goal[0]) ** 2 + (y - goal[1]) ** 2 + (z - goal[2]) ** 2)

    return V


@jit
def dV_pos_dx(z: Array, goal: Array) -> Array:
    """Jacobian for position goal function."""
    return jacfwd(V_pos)(z, goal)


@jit
def dV2_pos_dx2(z: Array, goal: Array) -> Array:
    """Hessian for position goal function."""
    return jacfwd(jacrev(V_pos))(z, goal)


def position(goal: Array) -> CertificateCollection:
    """Generates Lyapunov function for position convergence.

    Args:
        goal (Array): goal position in inertial frame

    Returns
    -------
        CertificateCollection: certificate functions
    """

    def v_func(t, x):
        return V_pos(jnp.hstack([x, t]), goal)  # type: ignore[return-value]

    def j_func(t, x):
        return dV_pos_dx(jnp.hstack([x, t]), goal)[:N]  # type: ignore[return-value]

    def h_func(t, x):
        return dV2_pos_dx2(jnp.hstack([x, t]), goal)[:N, :N]  # type: ignore[return-value]

    def p_func(t, x):
        return dV_pos_dx(jnp.hstack([x, t]), goal)[-1]  # type: ignore[return-value]

    return CertificateCollection(
        [v_func],
        [j_func],
        [h_func],
        [p_func],
        [],
    )


###############################################################################
#! Attitude Convergence
@jit
def V_att(state: Array, lam: float) -> Array:
    """Attitude convergence function (quadrotor is in hover-mode at goal location)."""
    x, y, z, _, _, _, phi, theta, psi, _, _, _, _ = state
    Rr = rotation_body_to_inertial(phi, theta, psi)
    V = 0.5 * jnp.linalg.norm(jnp.array([x, y, z]) + lam * jnp.matmul(Rr, jnp.array([1, 1, 1])))

    return V


@jit
def dV_att_dx(z: Array, lam: float) -> Array:
    """Jacobian for attitude convergence function."""
    return jacfwd(V_att)(z, lam)


@jit
def dV2_att_dx2(z: Array, lam: float) -> Array:
    """Hessian for attitude convergence function."""
    return jacfwd(jacrev(V_att))(z, lam)


def attitude(lam: float = 0.25) -> CertificateCollection:
    """Generates Lyapunov function for attitude convergence.

    Args:
        lam (float): length of lever arm

    Returns
    -------
        CertificateCollection: certificate functions
    """

    def v_func(t, x):
        return V_att(jnp.hstack([x, t]), lam)  # type: ignore[return-value]

    def j_func(t, x):
        return dV_att_dx(jnp.hstack([x, t]), lam)[:N]  # type: ignore[return-value]

    def h_func(t, x):
        return dV2_att_dx2(jnp.hstack([x, t]), lam)[:N, :N]  # type: ignore[return-value]

    def p_func(t, x):
        return dV_att_dx(jnp.hstack([x, t]), lam)[-1]  # type: ignore[return-value]

    return CertificateCollection(
        [v_func],
        [j_func],
        [h_func],
        [p_func],
        [],
    )


###############################################################################
#! Velocity Convergence
@jit
def V_vel(state: Array, control: Callable[[float, Array], Tuple[Array, Array, Array]]) -> Array:
    """Velocity convergence function (quadrotor has zero translational and rotational velocity)."""
    _, _, _, u, v, w, phi, theta, psi, p, q, r, _ = state
    Rr = rotation_body_to_inertial(phi, theta, psi)
    control_res = control(state)  # type: ignore

    vd = control_res[1]
    ad = control_res[2]

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
    """Jacobian for velocity convergence function."""
    return jacfwd(V_vel)(z)


@jit
def dV2_vel_dx2(z: Array) -> Array:
    """Hessian for velocity convergence function."""
    return jacfwd(jacrev(V_vel))(z)


def velocity(goal: Array = jnp.zeros(3)) -> CertificateCollection:
    """Generates Lyapunov function for velocity convergence (stub).

    Args:
        goal (Array): goal velocity and attitude

    Returns
    -------
        CertificateCollection: empty certificate collection
    """
    return EMPTY_CERTIFICATE_COLLECTION
