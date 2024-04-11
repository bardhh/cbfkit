from jax import Array
from cbfkit.utils.user_types import LyapunovTuple
from cbfkit.models.fixed_wing_uav.certificate_functions.lyapunov_function_catalog import (
    position_ff,
    position_order2,
    velocity,
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
    return ([lambda V: -c1 * V**e1 - c2 * V**e2 if V > 0 else 0],)


def fxts_lyapunovs_ff(
    goal: Array, T: float, c1: float, c2: float, e1: float, e2: float
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

    def funcs():
        return position_ff(goal, T) + fxts_lyapunov_conditions(c1, c2, e1, e2)

    return funcs


def fxts_lyapunovs_order2(goal: Array, c1: float, c2: float, e1: float, e2: float) -> LyapunovTuple:
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
        return position_order2(goal, c1, c2, e1, e2) + fxts_lyapunov_conditions(c1, c2, e1, e2)

    return funcs


def fxts_lyapunovs_vel(
    goal: Array,
    r: float,
    c1: float,
    c2: float,
    e1: float,
    e2: float,
) -> LyapunovTuple:
    """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

    Args:
        goal (Array): goal location
        R (float): radius (in m) around goal location defining goal set
        c1 (float): convergence constant 1
        c2 (float): convergence constant 2
        e1 (float): exponential constant 1
        e2 (float): exponential constant 2

    Returns:
        LyapunovTuple: all inforrmation needed for CLF constraint in QP
    """

    def funcs():
        return velocity(goal, r) + fxts_lyapunov_conditions(c1, c2, e1, e2)

    return funcs
