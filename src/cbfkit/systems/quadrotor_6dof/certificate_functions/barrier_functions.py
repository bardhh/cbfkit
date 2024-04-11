import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array
from typing import Optional
from cbfkit.utils.user_types import (
    BarrierCallable,
    BarrierJacobianCallable,
    BarrierHessianCallable,
    BarrierPartialCallable,
    BarrierTuple,
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


    Returns:
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

    Returns:
        ret (Array): value of Jacobian evaluated at time and state
    """
    return jacfwd(h_att)(z, att_limit)


@jit
def dh2_att_dx2(z: Array, att_limit: float) -> Array:
    """Hessian for attitude constraint function (prevent quadrotor from flipping over).

    Arguments:
        z (Array): concatenated time and state vector
        att_limit (float): attitude limit in rad

    Returns:
        ret (Array): value of Hessian evaluated at time and state
    """
    return jacfwd(jacrev(h_att))(z, att_limit)


def attitude(limit: float) -> BarrierTuple:
    b_func: BarrierCallable = lambda t, x: h_att(jnp.hstack([x, t]), limit)  # type: ignore[return-value]
    j_func: BarrierJacobianCallable = lambda t, x: dh_att_dx(jnp.hstack([x, t]), limit)[:N]  # type: ignore[return-value]
    h_func: BarrierHessianCallable = lambda t, x: dh2_att_dx2(jnp.hstack([x, t]), limit)[:N, :N]  # type: ignore[return-value]
    p_func: BarrierPartialCallable = lambda t, x: dh_att_dx(jnp.hstack([x, t]), limit)[-1]  # type: ignore[return-value]

    return (
        b_func,
        j_func,
        h_func,
        p_func,
    )


###############################################################################
#! Altitude Constraint
@jit
def h_alt(z: Array, alt_limit: float, k: Optional[float] = 2.0, n: Optional[int] = 2) -> Array:
    """Altitude constraint function (prevent quadrotor from crashing into ground or ceiling).

    Arguments:
        z (Array): concatenated time and state vector
        alt_limit (float): altitude limit in rad


    Returns:
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

    Returns:
        ret (Array): value of Jacobian evaluated at time and state
    """
    return jacfwd(h_alt)(z, alt_limit)


@jit
def dh2_alt_dx2(z: Array, alt_limit: float) -> Array:
    """Hessian for altitude constraint function (prevent quadrotor from crashing).

    Arguments:
        z (Array): concatenated time and state vector
        att_limit (float): attitude limit in rad

    Returns:
        ret (Array): value of Hessian evaluated at time and state
    """
    return jacfwd(jacrev(h_alt))(z, alt_limit)


def altitude(limit: float) -> BarrierTuple:
    b_func: BarrierCallable = lambda t, x: h_alt(jnp.hstack([x, t]), limit)  # type: ignore[return-value]
    j_func: BarrierJacobianCallable = lambda t, x: dh_alt_dx(jnp.hstack([x, t]), limit)[:N]  # type: ignore[return-value]
    h_func: BarrierHessianCallable = lambda t, x: dh2_alt_dx2(jnp.hstack([x, t]), limit)[:N, :N]  # type: ignore[return-value]
    p_func: BarrierPartialCallable = lambda t, x: dh_alt_dx(jnp.hstack([x, t]), limit)[-1]  # type: ignore[return-value]

    return (
        b_func,
        j_func,
        h_func,
        p_func,
    )


# ###############################################################################
# #! Circular Obstacle Avoidance (no prediction) -- h >= 0
# @jit
# def hfront(x: Array, cx: float, cy: float, r: float) -> Array:
#     """Obstacle avoidance constraint function for tractor/truck. Super-level set convention.

#     Args:
#         x (array-like): concatenated time and state vector
#         cx (float): x-coordinate of center of obstacle
#         cy (float): y-coordinate of center of obstacle
#         r (float): radius of obstacle

#     Returns:
#         ret (float): value of constraint function evaluated at time and state

#     """
#     xt = x[0] + L1 / 2 * jnp.cos(x[4])
#     yt = x[1] + L1 / 2 * jnp.sin(x[4])

#     return (xt - cx) ** 2 + (yt - cy) ** 2 - (r + L1 / 2 + CORNER_ADJUSTMENT) ** 2


# @jit
# def dhfrontdx(x: Array, cx: float, cy: float, r: float) -> Array:
#     return jacfwd(hfront)(x, cx, cy, r)


# @jit
# def d2hfrontdx2(x: Array, cx: float, cy: float, r: float) -> Array:
#     return jacrev(jacfwd(hfront))(x, cx, cy, r)


# @jit
# def hrear(x: Array, cx: float, cy: float, r: float, l2len: float) -> Array:
#     """Obstacle avoidance constraint function for tractor/truck. Super-level set convention.

#     Args:
#         x (array-like): concatenated time and state vector
#         cx (float): x-coordinate of center of obstacle
#         cy (float): y-coordinate of center of obstacle
#         r (float): radius of obstacle

#     Returns:
#         ret (float): value of constraint function evaluated at time and state

#     """
#     xt = x[0] - l2len * jnp.cos(x[5]) - LH * jnp.cos(x[4])
#     yt = x[1] - l2len * jnp.sin(x[5]) - LH * jnp.sin(x[4])

#     return (xt - cx) ** 2 + (yt - cy) ** 2 - (r + L2 / 4 + CORNER_ADJUSTMENT) ** 2


# @jit
# def dhreardx(x: Array, cx: float, cy: float, r: float, l2len: float) -> Array:
#     return jacfwd(hrear)(x, cx, cy, r, l2len)


# @jit
# def d2hreardx2(x: Array, cx: float, cy: float, r: float, l2len: float) -> Array:
#     return jacrev(jacfwd(hrear))(x, cx, cy, r, l2len)


# def circular_obstacle_front(cx: float, cy: float, r: float) -> BarrierTuple:
#     b_func: BarrierCallable = lambda t, x: hfront(jnp.hstack([x, t]), cx, cy, r)  # type: ignore[return-value]
#     j_func: BarrierJacobianCallable = lambda t, x: dhfrontdx(jnp.hstack([x, t]), cx, cy, r)[:N]  # type: ignore[return-value]
#     h_func: BarrierHessianCallable = lambda t, x: d2hfrontdx2(jnp.hstack([x, t]), cx, cy, r)[:N, :N]  # type: ignore[return-value]
#     p_func: BarrierPartialCallable = lambda t, x: dhfrontdx(jnp.hstack([x, t]), cx, cy, r)[-1]  # type: ignore[return-value]

#     return (
#         b_func,
#         j_func,
#         h_func,
#         p_func,
#     )


# def circular_obstacle_rear(cx: float, cy: float, r: float, l2len: float) -> BarrierTuple:
#     b_func: BarrierCallable = lambda t, x: hrear(jnp.hstack([x, t]), cx, cy, r, l2len) if x[1] > 3.0 else jnp.array([100.0])  # type: ignore[return-value]
#     j_func: BarrierJacobianCallable = lambda t, x: dhreardx(jnp.hstack([x, t]), cx, cy, r, l2len)[:N] if x[1] > 3.0 else jnp.zeros((N,))  # type: ignore[return-value]
#     h_func: BarrierHessianCallable = lambda t, x: d2hreardx2(jnp.hstack([x, t]), cx, cy, r, l2len)[:N, :N] if x[1] > 3.0 else jnp.zeros((N, N))  # type: ignore[return-value]
#     p_func: BarrierPartialCallable = lambda t, x: dhreardx(jnp.hstack([x, t]), cx, cy, r, l2len)[-1] if x[1] > 3.0 else jnp.array([0.0])  # type: ignore[return-value]

#     return (
#         b_func,
#         j_func,
#         h_func,
#         p_func,
#     )


# ###############################################################################
# #! Circular Obstacle Avoidance -- h + T * hdot >= 0
# @jit
# def hfrontt(x: Array, cx: float, cy: float, r: float, tfuture: float) -> Array:
#     """Obstacle avoidance constraint function for tractor/truck. Super-level set convention.

#     Args:
#         x (array-like): concatenated time and state vector
#         cx (float): x-coordinate of center of obstacle
#         cy (float): y-coordinate of center of obstacle
#         r (float): radius of obstacle

#     Returns:
#         ret (float): value of constraint function evaluated at time and state

#     """
#     xt = x[0] + L1 / 2 * jnp.cos(x[4])
#     yt = x[1] + L1 / 2 * jnp.sin(x[4])
#     theta1dot = x[2] / L1 * jnp.tan(x[6])
#     xdot = x[2] * jnp.cos(x[4]) - L1 / 2 * theta1dot * jnp.sin(x[4])
#     ydot = x[2] * jnp.sin(x[4]) + L1 / 2 * theta1dot * jnp.cos(x[4])

#     b = (xt - cx) ** 2 + (yt - cy) ** 2 - (r + L1 / 2 + CORNER_ADJUSTMENT) ** 2
#     bdot = 2 * (xt - cx) * xdot + 2 * (yt - cy) * ydot

#     return b + bdot * tfuture


# @jit
# def dhfronttdx(x: Array, cx: float, cy: float, r: float, tfuture: float) -> Array:
#     return jacfwd(hfrontt)(x, cx, cy, r, tfuture)


# @jit
# def d2hfronttdx2(x: Array, cx: float, cy: float, r: float, tfuture: float) -> Array:
#     return jacrev(jacfwd(hfrontt))(x, cx, cy, r, tfuture)


# @jit
# def hreart(x: Array, cx: float, cy: float, r: float, l2len: float, tfuture: float) -> Array:
#     """Obstacle avoidance constraint function for tractor/truck. Super-level set convention.

#     Args:
#         x (array-like): concatenated time and state vector
#         cx (float): x-coordinate of center of obstacle
#         cy (float): y-coordinate of center of obstacle
#         r (float): radius of obstacle

#     Returns:
#         ret (float): value of constraint function evaluated at time and state

#     """
#     xt = x[0] - l2len * jnp.cos(x[5]) - LH * jnp.cos(x[4])
#     yt = x[1] - l2len * jnp.sin(x[5]) - LH * jnp.sin(x[4])
#     theta1dot = x[2] / L1 * jnp.tan(x[6])
#     theta2dot = x[2] / L2 * (jnp.sin(x[6] - x[5]) - LH / L1 * jnp.cos(x[4] - x[5]) * jnp.tan(x[6]))
#     xdot = x[2] * jnp.cos(x[4]) + l2len * theta2dot * jnp.sin(x[5]) + LH * theta1dot * jnp.sin(x[4])
#     ydot = x[2] * jnp.sin(x[4]) - l2len * theta2dot * jnp.cos(x[5]) - LH * theta1dot * jnp.cos(x[4])

#     b = (xt - cx) ** 2 + (yt - cy) ** 2 - (r + L2 / 4 + CORNER_ADJUSTMENT) ** 2
#     bdot = 2 * (xt - cx) * xdot + 2 * (yt - cy) * ydot

#     return b + bdot * tfuture


# @jit
# def dhreartdx(x: Array, cx: float, cy: float, r: float, l2len: float, tfuture: float) -> Array:
#     return jacfwd(hreart)(x, cx, cy, r, l2len, tfuture)


# @jit
# def d2hreartdx2(x: Array, cx: float, cy: float, r: float, l2len: float, tfuture: float) -> Array:
#     return jacrev(jacfwd(hreart))(x, cx, cy, r, l2len, tfuture)


# def circular_obstacle_front_t(cx: float, cy: float, r: float, tfuture: float) -> BarrierTuple:
#     b_func: BarrierCallable = lambda t, x: hfrontt(jnp.hstack([x, t]), cx, cy, r, tfuture)  # type: ignore[return-value]
#     j_func: BarrierJacobianCallable = lambda t, x: dhfronttdx(jnp.hstack([x, t]), cx, cy, r, tfuture)[:N]  # type: ignore[return-value]
#     h_func: BarrierHessianCallable = lambda t, x: d2hfronttdx2(jnp.hstack([x, t]), cx, cy, r, tfuture)[:N, :N]  # type: ignore[return-value]
#     p_func: BarrierPartialCallable = lambda t, x: dhfronttdx(jnp.hstack([x, t]), cx, cy, r, tfuture)[-1]  # type: ignore[return-value]

#     return (
#         b_func,
#         j_func,
#         h_func,
#         p_func,
#     )


# def circular_obstacle_rear_t(
#     cx: float, cy: float, r: float, l2len: float, tfuture: float
# ) -> BarrierTuple:
#     b_func: BarrierCallable = lambda t, x: hreart(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)  # type: ignore[return-value]
#     j_func: BarrierJacobianCallable = lambda t, x: dhreartdx(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)[:N]  # type: ignore[return-value]
#     h_func: BarrierHessianCallable = lambda t, x: d2hreartdx2(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)[:N, :N]  # type: ignore[return-value]
#     p_func: BarrierPartialCallable = lambda t, x: dhreartdx(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)[-1]  # type: ignore[return-value]

#     return (
#         b_func,
#         j_func,
#         h_func,
#         p_func,
#     )


# ###############################################################################
# #! Circular Obstacle Avoidance -- Future-Focused CBF (Constant Velocity)
# @jit
# def hfrontffv(x: Array, cx: float, cy: float, r: float, tfuture: float) -> Array:
#     """Obstacle avoidance constraint function for tractor/truck. Super-level set convention.

#     Args:
#         x (array-like): concatenated time and state vector
#         cx (float): x-coordinate of center of obstacle
#         cy (float): y-coordinate of center of obstacle
#         r (float): radius of obstacle

#     Returns:
#         ret (float): value of constraint function evaluated at time and state

#     """
#     xt = x[0] + L1 / 2 * jnp.cos(x[4])
#     yt = x[1] + L1 / 2 * jnp.sin(x[4])
#     theta1dot = x[2] / L1 * jnp.tan(x[6])
#     xdot = x[2] * jnp.cos(x[4]) - L1 / 2 * theta1dot * jnp.sin(x[4])
#     ydot = x[2] * jnp.sin(x[4]) + L1 / 2 * theta1dot * jnp.cos(x[4])

#     cxdot = 0.0
#     cydot = 0.0

#     # FF-CBF
#     dx, dy, dvx, dvy = xt - cx, yt - cy, xdot - cxdot, ydot - cydot
#     tau_hat = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2 + 1e-3)
#     tau = sigmoid_func(tau_hat, tfuture)

#     return (
#         dx**2
#         + dy**2
#         + 2 * tau * (dx * dvx + dy * dvy)
#         + tau**2 * (dvx**2 + dvy**2)
#         - (r + L2 / 4 + CORNER_ADJUSTMENT) ** 2
#     )


# @jit
# def dhfrontffvdx(x: Array, cx: float, cy: float, r: float, tfuture: float) -> Array:
#     return jacfwd(hfrontffv)(x, cx, cy, r, tfuture)


# @jit
# def d2hfrontffvdx2(x: Array, cx: float, cy: float, r: float, tfuture: float) -> Array:
#     return jacrev(jacfwd(hfrontffv))(x, cx, cy, r, tfuture)


# @jit
# def hrearffv(x: Array, cx: float, cy: float, r: float, l2len: float, tfuture: float) -> Array:
#     """Obstacle avoidance constraint function for tractor/truck. Super-level set convention.

#     Args:
#         x (array-like): concatenated time and state vector
#         cx (float): x-coordinate of center of obstacle
#         cy (float): y-coordinate of center of obstacle
#         r (float): radius of obstacle

#     Returns:
#         ret (float): value of constraint function evaluated at time and state

#     """
#     xt = x[0] - l2len * jnp.cos(x[5]) - LH * jnp.cos(x[4])
#     yt = x[1] - l2len * jnp.sin(x[5]) - LH * jnp.sin(x[4])
#     theta1dot = x[2] / L1 * jnp.tan(x[6])
#     theta2dot = x[2] / L2 * (jnp.sin(x[4] - x[5]) - LH / L1 * jnp.cos(x[4] - x[5]) * jnp.tan(x[6]))
#     xdot = x[2] * jnp.cos(x[4]) + l2len * theta2dot * jnp.sin(x[5]) + LH * theta1dot * jnp.sin(x[4])
#     ydot = x[2] * jnp.sin(x[4]) - l2len * theta2dot * jnp.cos(x[5]) - LH * theta1dot * jnp.cos(x[4])

#     # FF-CBF
#     dx, dy, dvx, dvy = xt - cx, yt - cy, xdot, ydot
#     tau_hat = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2 + 1e-3)
#     tau = sigmoid_func(tau_hat, tfuture)

#     return (
#         dx**2
#         + dy**2
#         + 2 * tau * (dx * dvx + dy * dvy)
#         + tau**2 * (dvx**2 + dvy**2)
#         - (r + L2 / 4 + CORNER_ADJUSTMENT) ** 2
#     )


# @jit
# def dhrearffvdx(x: Array, cx: float, cy: float, r: float, l2len: float, tfuture: float) -> Array:
#     return jacfwd(hrearffv)(x, cx, cy, r, l2len, tfuture)


# @jit
# def d2hrearffvdx2(x: Array, cx: float, cy: float, r: float, l2len: float, tfuture: float) -> Array:
#     return jacrev(jacfwd(hrearffv))(x, cx, cy, r, l2len, tfuture)


# def circular_obstacle_front_ffv(cx: float, cy: float, r: float, tfuture: float) -> BarrierTuple:
#     b_func: BarrierCallable = lambda t, x: hfrontffv(jnp.hstack([x, t]), cx, cy, r, tfuture)  # type: ignore[return-value]
#     j_func: BarrierJacobianCallable = lambda t, x: dhfrontffvdx(jnp.hstack([x, t]), cx, cy, r, tfuture)[:N]  # type: ignore[return-value]
#     h_func: BarrierHessianCallable = lambda t, x: d2hfrontffvdx2(jnp.hstack([x, t]), cx, cy, r, tfuture)[:N, :N]  # type: ignore[return-value]
#     p_func: BarrierPartialCallable = lambda t, x: dhfrontffvdx(jnp.hstack([x, t]), cx, cy, r, tfuture)[-1]  # type: ignore[return-value]

#     return (
#         b_func,
#         j_func,
#         h_func,
#         p_func,
#     )


# def circular_obstacle_rear_ffv(
#     cx: float, cy: float, r: float, l2len: float, tfuture: float
# ) -> BarrierTuple:
#     b_func: BarrierCallable = lambda t, x: hrearffv(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)  # type: ignore[return-value]
#     j_func: BarrierJacobianCallable = lambda t, x: dhrearffvdx(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)[:N]  # type: ignore[return-value]
#     h_func: BarrierHessianCallable = lambda t, x: d2hrearffvdx2(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)[:N, :N]  # type: ignore[return-value]
#     p_func: BarrierPartialCallable = lambda t, x: dhrearffvdx(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)[-1]  # type: ignore[return-value]

#     return (
#         b_func,
#         j_func,
#         h_func,
#         p_func,
#     )


# ###############################################################################
# #! Circular Obstacle Avoidance -- Future-Focused CBF (Constant Acceleration)
# @jit
# def hfrontffa(x: Array, cx: float, cy: float, r: float, tfuture: float) -> Array:
#     """Obstacle avoidance constraint function for tractor/truck. Super-level set convention.

#     Args:
#         x (array-like): concatenated time and state vector
#         cx (float): x-coordinate of center of obstacle
#         cy (float): y-coordinate of center of obstacle
#         r (float): radius of obstacle

#     Returns:
#         ret (float): value of constraint function evaluated at time and state

#     """
#     xt = x[0] + L1 / 2 * jnp.cos(x[4])
#     yt = x[1] + L1 / 2 * jnp.sin(x[4])
#     theta1dot = x[2] / L1 * jnp.tan(x[6])
#     xdot = x[2] * jnp.cos(x[4]) - L1 / 2 * theta1dot * jnp.sin(x[4])
#     ydot = x[2] * jnp.sin(x[4]) + L1 / 2 * theta1dot * jnp.cos(x[4])
#     theta1ddot = (
#         x[3] / L1 * jnp.tan(x[6]) + x[2] / L1 / jnp.cos(x[6]) ** 2 * 0.0
#     )  # assume omega = 0
#     xddot = (
#         x[3] * jnp.cos(x[4])
#         - x[2] * theta1dot * jnp.sin(x[4])
#         - L1 / 2 * theta1ddot * jnp.sin(x[4])
#         - L1 / 2 * theta1dot**2 * jnp.cos(x[4])
#     )
#     yddot = (
#         x[3] * jnp.sin(x[4])
#         + x[2] * theta1dot * jnp.cos(x[4])
#         + L1 / 2 * theta1ddot * jnp.cos(x[4])
#         - L1 / 2 * theta1dot**2 * jnp.sin(x[4])
#     )
#     cxdot = 0.0
#     cydot = 0.0

#     # FF-CBF
#     dx, dy, dvx, dvy, dax, day = xt - cx, yt - cy, xdot - cxdot, ydot - cydot, xddot, yddot
#     tau_hat = compute_tau_jerk(dx, dy, dvx, dvy, dax, day)
#     tau = sigmoid_func(tau_hat, tfuture)

#     return (
#         (dx + dvx * tau + 1 / 2 * dax * tau**2) ** 2
#         + (dy + dvy * tau + 1 / 2 * day * tau**2) ** 2
#         - (r + L1 / 2 + CORNER_ADJUSTMENT) ** 2
#     )


# @jit
# def dhfrontffadx(x: Array, cx: float, cy: float, r: float, tfuture: float) -> Array:
#     return jacfwd(hfrontffa)(x, cx, cy, r, tfuture)


# @jit
# def d2hfrontffadx2(x: Array, cx: float, cy: float, r: float, tfuture: float) -> Array:
#     return jacrev(jacfwd(hfrontffa))(x, cx, cy, r, tfuture)


# @jit
# def hrearffa(x: Array, cx: float, cy: float, r: float, l2len: float, tfuture: float) -> Array:
#     """Obstacle avoidance constraint function for tractor/truck. Super-level set convention.

#     Args:
#         x (array-like): concatenated time and state vector
#         cx (float): x-coordinate of center of obstacle
#         cy (float): y-coordinate of center of obstacle
#         r (float): radius of obstacle

#     Returns:
#         ret (float): value of constraint function evaluated at time and state

#     """
#     xt = x[0] - l2len * jnp.cos(x[5]) - LH * jnp.cos(x[4])
#     yt = x[1] - l2len * jnp.sin(x[5]) - LH * jnp.sin(x[4])

#     x1dot = x[2] * jnp.cos(x[4])
#     y1dot = x[2] * jnp.sin(x[4])
#     theta1dot = x[2] / L1 * jnp.tan(x[6])
#     theta2dot = x[2] / L2 * (jnp.sin(x[4] - x[5]) - LH / L1 * jnp.cos(x[4] - x[5]) * jnp.tan(x[6]))
#     x1ddot = x[3] * jnp.cos(x[4]) - x[2] * theta1dot * jnp.sin(x[4])
#     y1ddot = x[3] * jnp.sin(x[4]) + x[2] * theta1dot * jnp.cos(x[4])
#     theta1ddot = (
#         x[3] / L1 * jnp.tan(x[6]) + x[2] / L1 / jnp.cos(x[6]) ** 2 * 0.0
#     )  # assume omega = 0
#     theta2ddot = x[3] / L2 * (
#         jnp.sin(x[4] - x[5]) - LH / L1 * jnp.cos(x[4] - x[5]) * jnp.tan(x[6])
#     ) + x[2] / L2 * (
#         (theta1dot - theta2dot) * jnp.cos(x[4] - x[5])
#         + LH / L1 * (theta1dot - theta2dot) * jnp.cos(x[4] - x[5])
#     )  # assume omega = 0

#     xtdot = x1dot + l2len * theta2dot * jnp.sin(x[5]) + LH * theta1dot * jnp.sin(x[4])
#     ytdot = y1dot - l2len * theta2dot * jnp.cos(x[5]) - LH * theta1dot * jnp.cos(x[4])

#     xtddot = (
#         x1ddot
#         + l2len * theta2ddot * jnp.sin(x[5])
#         + l2len * theta2dot**2 * jnp.cos(x[5])
#         + LH * theta1ddot * jnp.sin(x[4])
#         + LH * theta1dot**2 * jnp.cos(x[4])
#     )
#     ytddot = (
#         y1ddot
#         - l2len * theta2ddot * jnp.cos(x[5])
#         + l2len * theta2dot**2 * jnp.sin(x[5])
#         - LH * theta1ddot * jnp.cos(x[4])
#         + LH * theta1dot**2 * jnp.sin(x[4])
#     )
#     cxdot = 0.0
#     cydot = 0.0

#     # FF-CBF
#     dx, dy, dvx, dvy, dax, day = xt - cx, yt - cy, xtdot - cxdot, ytdot - cydot, xtddot, ytddot
#     tau_hat = compute_tau_jerk(dx, dy, dvx, dvy, dax, day)
#     tau = sigmoid_func(tau_hat, tfuture)

#     return (
#         (dx + dvx * tau + 1 / 2 * dax * tau**2) ** 2
#         + (dy + dvy * tau + 1 / 2 * day * tau**2) ** 2
#         - (r + L2 / 4 + CORNER_ADJUSTMENT) ** 2
#     )


# @jit
# def dhrearffadx(x: Array, cx: float, cy: float, r: float, l2len: float, tfuture: float) -> Array:
#     return jacfwd(hrearffa)(x, cx, cy, r, l2len, tfuture)


# @jit
# def d2hrearffadx2(x: Array, cx: float, cy: float, r: float, l2len: float, tfuture: float) -> Array:
#     return jacrev(jacfwd(hrearffa))(x, cx, cy, r, l2len, tfuture)


# def circular_obstacle_front_ffa(cx: float, cy: float, r: float, tfuture: float) -> BarrierTuple:
#     b_func: BarrierCallable = lambda t, x: hfrontffa(jnp.hstack([x, t]), cx, cy, r, tfuture)  # type: ignore[return-value]
#     j_func: BarrierJacobianCallable = lambda t, x: dhfrontffadx(jnp.hstack([x, t]), cx, cy, r, tfuture)[:N]  # type: ignore[return-value]
#     h_func: BarrierHessianCallable = lambda t, x: d2hfrontffadx2(jnp.hstack([x, t]), cx, cy, r, tfuture)[:N, :N]  # type: ignore[return-value]
#     p_func: BarrierPartialCallable = lambda t, x: dhfrontffadx(jnp.hstack([x, t]), cx, cy, r, tfuture)[-1]  # type: ignore[return-value]

#     return (
#         b_func,
#         j_func,
#         h_func,
#         p_func,
#     )


# def circular_obstacle_rear_ffa(
#     cx: float, cy: float, r: float, l2len: float, tfuture: float
# ) -> BarrierTuple:
#     b_func: BarrierCallable = lambda t, x: hrearffa(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)  # type: ignore[return-value]
#     j_func: BarrierJacobianCallable = lambda t, x: dhrearffadx(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)[:N]  # type: ignore[return-value]
#     h_func: BarrierHessianCallable = lambda t, x: d2hrearffadx2(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)[:N, :N]  # type: ignore[return-value]
#     p_func: BarrierPartialCallable = lambda t, x: dhrearffadx(jnp.hstack([x, t]), cx, cy, r, l2len, tfuture)[-1]  # type: ignore[return-value]

#     return (
#         b_func,
#         j_func,
#         h_func,
#         p_func,
#     )
