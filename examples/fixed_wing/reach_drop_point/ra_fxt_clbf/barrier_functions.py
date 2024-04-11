"""
Module docstring
"""
from typing import List
from jax import Array
from cbfkit.utils.user_types import BarrierTuple
from cbfkit.models.fixed_wing_uav.certificate_functions.barrier_function_catalog import obstacle_ff


# Future-Focused Obstacle Avoidance CBF
def obstacle_ff_barriers(
    obstacles: List[Array], r_obs: List[float], tfuture: float
) -> BarrierTuple:
    """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

    Args:
        obstacles (List[Array]): list of obstacle (center of volume) locations
        r_obs (List[float]): list of obstacle radii
        tfuture (float): maximum lookahead time

    Returns:
        LyapunovTuple: all inforrmation needed for CLF constraint in QP
    """
    barrier_functions = []
    barrier_jacobians = []
    barrier_hessians = []
    barrier_times = []
    for obstacle, r in zip(obstacles, r_obs):
        bf, bj, bh, bt = obstacle_ff(obstacle, r, tfuture)
        barrier_functions.append(bf)
        barrier_jacobians.append(bj)
        barrier_hessians.append(bh)
        barrier_times.append(bt)

    def funcs():
        return barrier_functions, barrier_jacobians, barrier_hessians, barrier_times

    return funcs
