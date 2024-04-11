from jax import Array, lax
from typing import List
from cbfkit.utils.user_types import LyapunovTuple
from cbfkit.models.fixed_wing_uav.certificate_functions.barrier_lyapunov_function_catalog import (
    velocity_with_obstacles,
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
    return (
        [
            lambda V: lax.cond(
                V > 0, lambda _fake: -c1 * V**e1 - c2 * V**e2, lambda _fake: 0.0, 0
            )
        ],
    )


def fxts_lyapunov_barrier_vel_and_obs(
    goal: Array,
    r: float,
    obstacles: List[Array],
    robs: List[float],
    alpha: float,
    c1: float,
    c2: float,
    e1: float,
    e2: float,
) -> LyapunovTuple:
    """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

    Args:
        goal (Array): goal location
        r (float): radius (in m) around goal location defining goal set
        obstacles (List[Array]): list of locations of obstacles
        robs (List[float]): list of radii of obstacles
        tfuture (float): maximum lookahead time
        c1 (float): convergence constant 1
        c2 (float): convergence constant 2
        e1 (float): exponential constant 1
        e2 (float): exponential constant 2

    Returns:
        LyapunovTuple: all inforrmation needed for CLF constraint in QP
    """

    def funcs():
        return velocity_with_obstacles(goal, r, obstacles, robs, alpha) + fxts_lyapunov_conditions(
            c1, c2, e1, e2
        )

    return funcs


# def fxts_lyapunov_barrier_velpos(
#     goal: Array,
#     r: float,
#     obstacles: List[Array],
#     robs: List[float],
#     c1: float,
#     c2: float,
#     e1: float,
#     e2: float,
# ) -> LyapunovTuple:
#     """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

#     Args:
#         goal (Array): goal location
#         r (float): radius (in m) around goal location defining goal set
#         obstacles (List[Array]): list of locations of obstacles
#         robs (List[float]): list of radii of obstacles
#         tfuture (float): maximum lookahead time
#         c1 (float): convergence constant 1
#         c2 (float): convergence constant 2
#         e1 (float): exponential constant 1
#         e2 (float): exponential constant 2

#     Returns:
#         LyapunovTuple: all inforrmation needed for CLF constraint in QP
#     """

#     def funcs():
#         return velpos_with_obstacles(goal, r, obstacles, robs) + fxts_lyapunov_conditions(
#             c1, c2, e1, e2
#         )

#     return funcs
