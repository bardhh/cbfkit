"""Module docstring."""

from typing import Callable, List

from jax import Array
import jax.numpy as jnp
from cbfkit.certificates import certificate_package
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.utils.user_types import CertificateCollection


# Local 2D Barrier Function Factory
def cbf(obstacle: Array, r: float, tfuture: float):
    """
    Factory for simple distance based barrier: h(x) = (x-ox)^2 + (y-oy)^2 - r^2
    Ignores tfuture for now as VDP is 2D position.
    """
    def func(state: Array) -> Array:
        x, y = state[0], state[1]
        ox, oy = obstacle[0], obstacle[1]
        return (x - ox)**2 + (y - oy)**2 - r**2
    return func

obstacle_ff = certificate_package(cbf, n=2)


# Future-Focused Obstacle Avoidance CBF
def obstacle_ff_barriers(
    obstacles: List[Array], r_obs: List[float], tfuture: float
) -> Callable[[], CertificateCollection]:
    """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

    Args:
        obstacles (List[Array]): list of obstacle (center of volume) locations
        r_obs (List[float]): list of obstacle radii
        tfuture (float): maximum lookahead time

    Returns
    -------
        Callable[[], CertificateCollection]: all inforrmation needed for CLF constraint in QP
    """
    barrier_functions = []
    barrier_jacobians = []
    barrier_hessians = []
    barrier_times = []
    barrier_conditions = []
    for obstacle, r in zip(obstacles, r_obs):
        bf, bj, bh, bt, bc = obstacle_ff(certificate_conditions=linear_class_k(1.0), obstacle=obstacle, r=r, tfuture=tfuture)
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
