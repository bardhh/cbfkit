"""
ellipsoidal_obstacle.py

Contains functions defining a CBF for collision avoidance wrt an ellipsoidal obstacle.

Exportable:
    obstacle_ca

"""
from jax import jit, jacfwd, jacrev, Array
from cbfkit.controllers.utils.certificate_packager import certificate_package

N = 3


def cbf(obstacle: Array, ellipsoid: Array) -> Array:
    """Obstacle avoidance constraint function for approximate unicycle.
    Super-level set convention.

    Args:
        x (array-like): concatenated time and state vector
        obstacle (Array): x, y, t state of obstacle
        ellipsoid (Array): list of 2D ellipsoid parameters

    Returns:
        ret (float): value of constraint function evaluated at time and state
    """

    @jit
    def func(state_and_time: Array) -> Array:
        """ """
        x_e, y_e, _theta_e, _t = state_and_time
        x_o, y_o, _t = obstacle
        a1, a2 = ellipsoid

        b = ((x_e - x_o) / a1) ** 2 + ((y_e - y_o) / a2) ** 2 - 1.0

        return b

    return func


def cbf_grad(obstacle: Array, ellipsoid: Array) -> Array:
    """Jacobian for obstacle avoidance constraint function for approximate unicycle.
    Super-level set convention.

    Args:
        x (array-like): concatenated time and state vector
        obstacle (Array): x, y, t state of obstacle
        ellipsoid (Array): list of 2D ellipsoid parameters

    Returns:
        ret (float): value of constraint function evaluated at time and state
    """
    jacobian = jacfwd(cbf(obstacle, ellipsoid))

    @jit
    def func(state_and_time: Array) -> Array:
        """ """
        return jacobian(state_and_time)

    return func


def cbf_hess(obstacle: Array, ellipsoid: Array) -> Array:
    """Hessian for obstacle avoidance constraint function for approximate unicycle.
    Super-level set convention.

    Args:
        x (array-like): concatenated time and state vector
        obstacle (Array): x, y, t state of obstacle
        ellipsoid (Array): list of 2D ellipsoid parameters

    Returns:
        ret (float): value of constraint function evaluated at time and state
    """
    hessian = jacrev(jacfwd(cbf(obstacle, ellipsoid)))

    @jit
    def func(state_and_time: Array) -> Array:
        """ """
        return hessian(state_and_time)

    return func


###############################################################################
# 2nd Order CBF
###############################################################################
obstacle_ca = certificate_package(cbf, cbf_grad, cbf_hess, N)


# CX1, CY1, R1 = 1.0, 2.0, 0.5
# obstacle_avoidance_bf_1 = lambda t, x: h(jnp.hstack([x, t]), CX1, CY1, R1)
# obstacle_avoidance_bj_1 = lambda t, x: dhdx(jnp.hstack([x, t]), CX1, CY1, R1)[:N]
# obstacle_avoidance_bh_1 = lambda t, x: d2hdx2(jnp.hstack([x, t]), CX1, CY1, R1)[:N, :N]
# CX2, CY2, R2 = 2.0, 2.0, 0.5
# obstacle_avoidance_bf_2 = lambda t, x: h(jnp.hstack([x, t]), CX2, CY2, R2)
# obstacle_avoidance_bj_2 = lambda t, x: dhdx(jnp.hstack([x, t]), CX2, CY2, R2)[:N]
# obstacle_avoidance_bh_2 = lambda t, x: d2hdx2(jnp.hstack([x, t]), CX2, CY2, R2)[:N, :N]
# CX3, CY3, R3 = 0.0, 3.0, 0.5
# obstacle_avoidance_bf_3 = lambda t, x: h(jnp.hstack([x, t]), CX3, CY3, R3)
# obstacle_avoidance_bj_3 = lambda t, x: dhdx(jnp.hstack([x, t]), CX3, CY3, R3)[:N]
# obstacle_avoidance_bh_3 = lambda t, x: d2hdx2(jnp.hstack([x, t]), CX3, CY3, R3)[:N, :N]


# #! Accessible Objects
# barrier_functions = [
#     obstacle_avoidance_bf_1,
#     obstacle_avoidance_bf_2,
#     obstacle_avoidance_bf_3,
# ]

# barrier_jacobians = [
#     obstacle_avoidance_bj_1,
#     obstacle_avoidance_bj_2,
#     obstacle_avoidance_bj_3,
# ]

# barrier_hessians = [
#     obstacle_avoidance_bh_1,
#     obstacle_avoidance_bh_2,
#     obstacle_avoidance_bh_3,
# ]

# barrier_times = [
#     lambda t, x: 0,
#     lambda t, x: 0,
#     lambda t, x: 0,
# ]


# def barrier_funcs():
#     return barrier_functions, barrier_jacobians, barrier_hessians, barrier_times


# CX = [CX1, CX2, CX3]
# CY = [CY1, CY2, CY3]
# R = [R1, R2, R3]
