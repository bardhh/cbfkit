"""
velocity_with_obstacles.py

"""
from typing import List
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array
from cbfkit.controllers.utils.certificate_packager import certificate_package
from cbfkit.systems.fixed_wing_uav.models.beard2014_kinematic.certificate_functions.lyapunov_functions.velocity import (
    clf,
)
from cbfkit.systems.fixed_wing_uav.models.beard2014_kinematic.certificate_functions.barrier_functions.obstacle_avoidance.high_order import (
    cbf,
)


# constants
N = 6  # number of states


###############################################################################
## Goal velocity while avoiding obstacles
@jit
def cblf(
    state: Array,
    goal: Array,
    rg: float,
    obstacles: List[Array],
    robs: List[float],
    alpha: float,
) -> Array:
    """Barrier Lyapunov function for driving the vehicle to goal velocities
    while avoiding obstacles.

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal pos/vel vector [vg, yg, zg]
        rg (float): acceptable radius (in m) around goal location
        obstacles (List): list of arrays denoting obstacle (x,y,z) positions
        robs (List): list of floats denoting obstacle radii
        alpha (float): class K function for 2nd order CBF

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    # Generate current LBF value
    barriers = jnp.array([cbf(state, obs, ro, alpha) for (obs, ro) in zip(obstacles, robs)])
    V = clf(state, goal, rg) / (1 - jnp.sum(1 / (barriers)))

    return V


@jit
def cblf_grad(
    state: Array,
    goal: Array,
    rg: float,
    obstacles: List[Array],
    robs: List[float],
    alpha: float,
) -> Array:
    """Jacobian for Barrier-Lyapunov function for driving the vehicle to goal velocities
    while avoiding obstacles.

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal pos/vel vector [vg, yg, zg]
        rg (float): acceptable radius (in m) around goal location
        obstacles (List): list of arrays denoting obstacle (x,y,z) positions
        robs (List): list of floats denoting obstacle radii
        alpha (float): class K function for 2nd order CBF

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(cblf)(state, goal, rg, obstacles, robs, alpha)


@jit
def cblf_hess(
    state: Array,
    goal: Array,
    rg: float,
    obstacles: List[Array],
    robs: List[float],
    alpha: float,
) -> Array:
    """Hessian for Barrier-Lyapunov function for driving the vehicle to goal velocities
    while avoiding obstacles.

    Args:
        state (Array): concatenated time and state vector
        goal (Array): goal pos/vel vector [vg, yg, zg]
        rg (float): acceptable radius (in m) around goal location
        obstacles (List): list of arrays denoting obstacle (x,y,z) positions
        robs (List): list of floats denoting obstacle radii
        alpha (float): class K function for 2nd order CBF

    Returns:
        ret (float): value of goal function evaluated at time and state
    """
    return jacfwd(jacrev(cblf))(state, goal, rg, obstacles, robs, alpha)


###############################################################################
# CBLF Package: velocity tracking + ho-cbf obstacle avoidance
###############################################################################
velocity_with_obstacles = certificate_package(cblf, cblf_grad, cblf_hess, N)


# def velocity_with_obstacles(
#     goal: Array,
#     rg: float,
#     obstacles: List[Array],
#     robs: List[float],
#     alpha: float,
# ) -> LyapunovTuple:
#     """Callable that generates Lyapunov function and its associated

#     Args:
#         goal (Array): goal pos/vel vector [vg, yg, zg]
#         rg (float): acceptable radius (in m) around goal location
#         obstacles (List): list of arrays denoting obstacle (x,y,z) positions
#         robs (List): list of floats denoting obstacle radii
#         alpha (float): class K function for 2nd order CBF

#     Returns:
#         LyapunovTuple: _description_
#     """
#     v_func: LyapunovCallable = lambda t, x: V_velandobs2(jnp.hstack([x, t]), goal, rg, obstacles, robs, alpha)  # type: ignore[return-value]
#     j_func: LyapunovJacobianCallable = lambda t, x: dV_velandobs2_dx(jnp.hstack([x, t]), goal, rg, obstacles, robs, alpha)[:N]  # type: ignore[return-value]
#     h_func: LyapunovHessianCallable = lambda t, x: dV2_velandobs2_dx2(jnp.hstack([x, t]), goal, rg, obstacles, robs, alpha)[:N, :N]  # type: ignore[return-value]
#     p_func: LyapunovPartialCallable = lambda t, x: dV_velandobs2_dx(jnp.hstack([x, t]), goal, rg, obstacles, robs, alpha)[-1]  # type: ignore[return-value]

#     return (
#         [v_func],
#         [j_func],
#         [h_func],
#         [p_func],
#     )
