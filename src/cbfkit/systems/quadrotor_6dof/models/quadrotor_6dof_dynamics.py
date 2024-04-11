import jax.numpy as jnp
from jax import jit, Array
from typing import Optional, Union
from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns

g_accel = 9.81


def quadrotor_6dof_dynamics(
    m: float = 0.25,
    jx: float = 5e-3,
    jy: float = 5e-3,
    jz: float = 5e-3,
    sigma: Optional[Union[Array, None]] = None,
) -> DynamicsCallable:
    """
    Returns a function that represents the 6 degree-of-freedom quadotor dynamics,
    which computes the drift vector 'f', control matrix 'g', and diffusion matrix
    's' (the argument sigma) based on the given state.

    Args:
        m (float): mass of quadrotor in kg
        jx (float): moment of inertia in principal body x direction
        jy (float): moment of inertia in principal body y direction
        jz (float): moment of inertia in principal body z direction
        sigma (Optional, Array): diffusion term in stochastic differential equation

    Returns:
        dynamics (Callable): takes state as input and returns dynamics components
            f, g, and s of the form dx = (f(x) + g(x)u)dt + s(x)dw

    """
    if sigma is not None:
        s = sigma
    else:
        s = jnp.zeros((12, 12))

    @jit
    def dynamics(x: Array) -> DynamicsCallableReturns:
        """
        Computes the drift vector 'f' and control matrix 'g' based on the given state x,
        which consists of pn, pe, h (positions in m), u, v, w (body-fixed velocities in m/s),
        phi, theta, psi (yaw, pitch, and roll angles in rad), and p, q, r (yaw, pitch, and
        roll rates in rad/s).

        Args:
            x (Array): state vector

        Returns:
            f, g, s (Tuple of Arrays): drift vector f, control matrix g, diffusion matrix s

        """
        nonlocal s

        _, _, _, u, v, w, phi, theta, psi, p, q, r = x
        f1 = jnp.hstack(
            [
                jnp.matmul(rotation_body_to_inertial(phi, theta, psi), jnp.array([u, v, w])),
                jnp.array([r * v - q * w - g_accel * jnp.sin(theta)]),
                jnp.array([p * w - r * u + g_accel * jnp.cos(theta) * jnp.sin(phi)]),
                jnp.array([q * u - p * v + g_accel * jnp.cos(theta) * jnp.cos(phi)]),
            ]
        )

        f2 = jnp.array(
            [
                p + q * jnp.sin(phi) * jnp.tan(theta) + r * jnp.cos(phi) * jnp.tan(theta),
                q * jnp.cos(phi) - r * jnp.sin(phi),
                q * jnp.sin(phi) / jnp.cos(theta) + r * jnp.cos(phi) / jnp.cos(theta),
                q * r * (jy - jz) / jx,
                p * r * (jz - jx) / jy,
                p * q * (jx - jy) / jz,
            ]
        )
        f = jnp.hstack([f1, f2])
        g = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-1 / m, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1 / jx, 0.0, 0.0],
                [0.0, 0.0, 1 / jy, 0.0],
                [0.0, 0.0, 0.0, 1 / jz],
            ]
        )

        return f, g, s

    return dynamics


def rotation_body_to_inertial(phi: float, theta: float, psi: float) -> Array:
    """Computes rotation matrix from body-fixed frame to inertial frame.

    Args:
        phi (float): roll Euler angle (rad)
        theta (float): pitch Euler angle (rad)
        psi (float): yaw Euler angle (rad)

    Returns:
        Rr: rotation matric
    """
    return jnp.array(
        [
            [
                (jnp.cos(theta) * jnp.cos(psi)),
                (jnp.sin(phi) * jnp.sin(theta) * jnp.cos(psi) - jnp.cos(phi) * jnp.sin(psi)),
                (jnp.cos(phi) * jnp.sin(theta) * jnp.cos(psi) + jnp.sin(phi) * jnp.sin(psi)),
            ],
            [
                (jnp.cos(theta) * jnp.sin(psi)),
                (jnp.sin(phi) * jnp.sin(theta) * jnp.sin(psi) + jnp.cos(phi) * jnp.cos(psi)),
                (jnp.cos(phi) * jnp.sin(theta) * jnp.sin(psi) - jnp.sin(phi) * jnp.cos(psi)),
            ],
            [jnp.sin(theta), -jnp.sin(phi) * jnp.cos(theta), -jnp.cos(phi) * jnp.cos(theta)],
        ]
    )
