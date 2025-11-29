"""
Module docstring
"""

from typing import Callable, List

from jax import Array

from cbfkit.systems.fixed_wing_uav.models.beard2014_kinematic.certificates.barrier_functions import (
    obstacle_ff,
)
from cbfkit.utils.user_types import CertificateCollection


# Future-Focused Obstacle Avoidance CBF
def obstacle_ff_barriers(
    obstacles: List[Array], r_obs: List[float], tfuture: float
) -> Callable[[], CertificateCollection]:
    """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

    Args:
        obstacles (List[Array]): list of obstacle (center of volume) locations
        r_obs (List[float]): list of obstacle radii
        tfuture (float): maximum lookahead time

    Returns:
        Callable[[], CertificateCollection]: all inforrmation needed for CLF constraint in QP
    """
    barrier_functions = []
    barrier_jacobians = []
    barrier_hessians = []
    barrier_times = []
    barrier_conditions = []
    for obstacle, r in zip(obstacles, r_obs):
        bf, bj, bh, bt, bc = obstacle_ff(obstacle, r, tfuture)
        barrier_functions.extend(bf)
        barrier_jacobians.extend(bj)
        barrier_hessians.extend(bh)
        barrier_times.extend(bt)
        barrier_conditions.extend(bc)

    def funcs():
        return (
            barrier_functions,
            barrier_jacobians,
            barrier_hessians,
            barrier_times,
            barrier_conditions,
        )

    return funcs
