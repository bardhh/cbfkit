from typing import List, Optional, Tuple

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit

from cbfkit.utils.user_types import (
    CertificateCallable,
    CertificateHessianCallable,
    CertificateJacobianCallable,
    CertificatePartialCallable,
)

# number of states
N = 12


###############################################################################
#! Attitude Constraint
@jit
def h_att(z: Array, att_limit: float, k: Optional[float] = 2.0) -> Array:
    """Attitude constraint function (prevent quadrotor from flipping over).

    Arguments:
        z (Array): concatenated time and state vector
        att_limit (float): attitude limit in rad


    Returns
    -------
        ret (float): value of constraint function evaluated at time and state
    """
    # HO-CBF requires state model
    _, _, _, _, _, _, phi, theta, _, p, q, r, _ = z
    phi_dot = p + q * jnp.sin(phi) * jnp.tan(theta) + r * jnp.cos(phi) * jnp.tan(theta)
    theta_dot = q * jnp.cos(phi) - r * jnp.sin(phi)

    h = jnp.cos(phi) * jnp.cos(theta) - jnp.cos(att_limit)
    h_dot = -phi_dot * jnp.sin(phi) * jnp.cos(theta) - theta_dot * jnp.sin(theta) * jnp.cos(phi)

    return h_dot + k * h


@jit
def dh_att_dx(z: Array, att_limit: float) -> Array:
    """Jacobian for attitude constraint function (prevent quadrotor from flipping over).

    Arguments:
        z (Array): concatenated time and state vector
        att_limit (float): attitude limit in rad

    Returns
    -------
        ret (Array): value of Jacobian evaluated at time and state
    """
    return jacfwd(h_att)(z, att_limit)


@jit
def dh2_att_dx2(z: Array, att_limit: float) -> Array:
    """Hessian for attitude constraint function (prevent quadrotor from flipping over).

    Arguments:
        z (Array): concatenated time and state vector
        att_limit (float): attitude limit in rad

    Returns
    -------
        ret (Array): value of Hessian evaluated at time and state
    """
    return jacfwd(jacrev(h_att))(z, att_limit)


def attitude(
    limit: float,
) -> Tuple[
    List[CertificateCallable],
    List[CertificateJacobianCallable],
    List[CertificateHessianCallable],
    List[CertificatePartialCallable],
]:
    """Provides a certificate for attitude constraints.

    Args:
        limit (float): The attitude limit.

    Returns:
        A tuple of lists containing the barrier function, its Jacobian, Hessian, and partial derivative.
    """

    def b_func(t, x):
        return h_att(jnp.hstack([x, t]), limit)  # type: ignore[return-value]

    def j_func(t, x):
        return dh_att_dx(jnp.hstack([x, t]), limit)[:N]  # type: ignore[return-value]

    def h_func(t, x):
        return dh2_att_dx2(jnp.hstack([x, t]), limit)[:N, :N]  # type: ignore[return-value]

    def p_func(t, x):
        return dh_att_dx(jnp.hstack([x, t]), limit)[-1]  # type: ignore[return-value]

    return (
        [b_func],
        [j_func],
        [h_func],
        [p_func],
    )


###############################################################################
#! Altitude Constraint
@jit
def h_alt(z: Array, alt_limit: float, k: float = 2.0, n: int = 2) -> Array:
    """Altitude constraint function (prevent quadrotor from crashing into ground or ceiling).

    Arguments:
        z (Array): concatenated time and state vector
        alt_limit (float): altitude limit in rad


    Returns
    -------
        ret (float): value of constraint function evaluated at time and state
    """
    # HO-CBF requires state model
    _, _, a, u, v, w, phi, theta, _, _, _, _, _ = z
    a_dot = (
        u * jnp.sin(theta) - v * jnp.sin(phi) * jnp.cos(theta) - w * jnp.cos(phi) * jnp.cos(theta)
    )

    h = 1 - ((a - alt_limit / 2) / (alt_limit / 2)) ** n
    h_dot = -n * ((a - alt_limit / 2) / (alt_limit / 2)) ** (n - 1) * a_dot

    return h_dot + k * h


@jit
def dh_alt_dx(z: Array, alt_limit: float) -> Array:
    """Jacobian for altitude constraint function (prevent quadrotor from crashing).

    Arguments:
        z (Array): concatenated time and state vector
        att_limit (float): attitude limit in rad

    Returns
    -------
        ret (Array): value of Jacobian evaluated at time and state
    """
    return jacfwd(h_alt)(z, alt_limit)


@jit
def dh2_alt_dx2(z: Array, alt_limit: float) -> Array:
    """Hessian for altitude constraint function (prevent quadrotor from crashing).

    Arguments:
        z (Array): concatenated time and state vector
        att_limit (float): attitude limit in rad

    Returns
    -------
        ret (Array): value of Hessian evaluated at time and state
    """
    return jacfwd(jacrev(h_alt))(z, alt_limit)


def altitude(
    limit: float,
) -> Tuple[
    List[CertificateCallable],
    List[CertificateJacobianCallable],
    List[CertificateHessianCallable],
    List[CertificatePartialCallable],
]:
    """Provides a certificate for altitude constraints.

    Args:
        limit (float): The altitude limit.

    Returns:
        A tuple of lists containing the barrier function, its Jacobian, Hessian, and partial derivative.
    """

    def b_func(t, x):
        return h_alt(jnp.hstack([x, t]), limit)  # type: ignore[return-value]

    def j_func(t, x):
        return dh_alt_dx(jnp.hstack([x, t]), limit)[:N]  # type: ignore[return-value]

    def h_func(t, x):
        return dh2_alt_dx2(jnp.hstack([x, t]), limit)[:N, :N]  # type: ignore[return-value]

    def p_func(t, x):
        return dh_alt_dx(jnp.hstack([x, t]), limit)[-1]  # type: ignore[return-value]

    return (
        [b_func],
        [j_func],
        [h_func],
        [p_func],
    )

