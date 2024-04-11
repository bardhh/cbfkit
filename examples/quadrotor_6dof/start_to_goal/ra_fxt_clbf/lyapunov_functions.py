import jax.numpy as jnp
from jax import Array
from control import lqr
from cbfkit.utils.user_types import LyapunovTuple
from cbfkit.models.quadrotor_6dof.certificate_functions.lyapunov_functions import (
    composite,
    geometric,
)


def fxts_lyapunov_conditions(c1: float, c2: float, e1: float, e2: float):
    """Generates function for computing RHS of Lyapunov conditions for FxTS.

    Args:
        c1 (float): convergence constant 1
        c2 (float): convergence constant 2
        e1 (float): exponential constant 1
        e2 (float): exponential constant 2

    Returns:
        callable[float]: FxTS Lyapunov conditions
    """
    return ([lambda V: -c1 * V**e1 - c2 * V**e2],)


def fxts_lyapunovs(goal: Array, c1: float, c2: float, e1: float, e2: float) -> LyapunovTuple:
    """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

    Args:
        goal (Array): goal location
        c1 (float): convergence constant 1
        c2 (float): convergence constant 2
        e1 (float): exponential constant 1
        e2 (float): exponential constant 2

    Returns:
        LyapunovTuple: all inforrmation needed for CLF constraint in QP
    """

    def funcs():
        return composite(goal) + fxts_lyapunov_conditions(c1, c2, e1, e2)

    return funcs


def fxts_geometric_lyapunovs(
    goal: Array,
    m: float,
    c1: float,
    c2: float,
    e1: float,
    e2: float,
    kx: float = 1.0,
    kv: float = 1.0,
) -> LyapunovTuple:
    """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

    Args:
        goal (Array): goal location
        c1 (float): convergence constant 1
        c2 (float): convergence constant 2
        e1 (float): exponential constant 1
        e2 (float): exponential constant 2

    Returns:
        LyapunovTuple: all inforrmation needed for CLF constraint in QP
    """
    A = jnp.zeros((6, 6))
    A = A.at[:3, 3:6].set(jnp.eye(3))
    B = jnp.zeros((6, 3))
    B = B.at[3:, :].set(jnp.eye(3))
    Q = jnp.eye(6)
    R = jnp.eye(3)

    # Compute LQR gain
    k_lqr, _, _ = lqr(A, B, Q, R)

    def funcs():
        return geometric(goal, k_lqr[:, :3], m, kx, kv) + fxts_lyapunov_conditions(c1, c2, e1, e2)

    return funcs
