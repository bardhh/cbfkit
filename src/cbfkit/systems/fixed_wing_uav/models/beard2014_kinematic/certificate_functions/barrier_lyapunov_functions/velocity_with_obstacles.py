"""
velocity_with_obstacles.py

"""

from typing import List
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
)
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
def cblf(
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
    barriers = [cbf(obs, ro, alpha) for (obs, ro) in zip(obstacles, robs)]
    control_lyap = clf(goal, rg)

    @jit
    def func(state: Array) -> Array:
        v = control_lyap(state)
        b = jnp.array([b(state) for b in barriers])

        return v / (1 - jnp.sum(jnp.exp(-b)))

    return func


def cblf_grad(
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
    jacobian = jacfwd(cblf(goal, rg, obstacles, robs, alpha))

    @jit
    def func(state):
        return jacobian(state)

    return func


def cblf_hess(
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
    hessian = jacfwd(jacrev(cblf(goal, rg, obstacles, robs, alpha)))

    @jit
    def func(state: Array) -> Array:
        return hessian(state)

    return func


###############################################################################
# CBLF Package: velocity tracking + ho-cbf obstacle avoidance
###############################################################################
velocity_with_obstacles = certificate_package(cblf, cblf_grad, cblf_hess, N)
