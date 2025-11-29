from typing import Callable, List

from jax import Array, lax

from cbfkit.systems.fixed_wing_uav.models.beard2014_kinematic.certificates.barrier_lyapunov_functions import (
    velocity_with_obstacles,
)
from cbfkit.utils.user_types import CertificateCollection, LyapunovTuple


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
        [lambda V: lax.cond(V > 0, lambda _fake: -c1 * V**e1 - c2 * V**e2, lambda _fake: 0.0, 0.0)],
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
) -> Callable[[], CertificateCollection]:
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
        cert_collection = velocity_with_obstacles(goal, r, obstacles, robs, alpha)
        conditions = fxts_lyapunov_conditions(c1, c2, e1, e2)
        return (
            cert_collection[0],
            cert_collection[1],
            cert_collection[2],
            cert_collection[3],
            cert_collection[4] + conditions[0],
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
