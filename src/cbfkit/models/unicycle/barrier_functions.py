import jax.numpy as jnp
from jax import jit, jacfwd, jacrev

#! Circular Obstacle Avoidance
N = 3


@jit
def h(x, cx, cy, r):
    return (x[0] - cx) ** 2 + (x[1] - cy) ** 2 - r**2


@jit
def dhdx(x, cx, cy, r):
    return jacfwd(h)(x, cx, cy, r)


@jit
def d2hdx2(x, cx, cy, r):
    return jacrev(jacfwd(h))(x, cx, cy, r)


CX1, CY1, R1 = 1.0, 2.0, 0.5
obstacle_avoidance_bf_1 = lambda t, x: h(jnp.hstack([x, t]), CX1, CY1, R1)
obstacle_avoidance_bj_1 = lambda t, x: dhdx(jnp.hstack([x, t]), CX1, CY1, R1)[:N]
obstacle_avoidance_bh_1 = lambda t, x: d2hdx2(jnp.hstack([x, t]), CX1, CY1, R1)[:N, :N]
CX2, CY2, R2 = 2.0, 2.0, 0.5
obstacle_avoidance_bf_2 = lambda t, x: h(jnp.hstack([x, t]), CX2, CY2, R2)
obstacle_avoidance_bj_2 = lambda t, x: dhdx(jnp.hstack([x, t]), CX2, CY2, R2)[:N]
obstacle_avoidance_bh_2 = lambda t, x: d2hdx2(jnp.hstack([x, t]), CX2, CY2, R2)[:N, :N]
CX3, CY3, R3 = 0.0, 3.0, 0.5
obstacle_avoidance_bf_3 = lambda t, x: h(jnp.hstack([x, t]), CX3, CY3, R3)
obstacle_avoidance_bj_3 = lambda t, x: dhdx(jnp.hstack([x, t]), CX3, CY3, R3)[:N]
obstacle_avoidance_bh_3 = lambda t, x: d2hdx2(jnp.hstack([x, t]), CX3, CY3, R3)[:N, :N]


#! Accessible Objects
barrier_functions = [
    obstacle_avoidance_bf_1,
    obstacle_avoidance_bf_2,
    obstacle_avoidance_bf_3,
]

barrier_jacobians = [
    obstacle_avoidance_bj_1,
    obstacle_avoidance_bj_2,
    obstacle_avoidance_bj_3,
]

barrier_hessians = [
    obstacle_avoidance_bh_1,
    obstacle_avoidance_bh_2,
    obstacle_avoidance_bh_3,
]

barrier_times = [
    lambda t, x: 0,
    lambda t, x: 0,
    lambda t, x: 0,
]


def barrier_funcs():
    return barrier_functions, barrier_jacobians, barrier_hessians, barrier_times


CX = [CX1, CX2, CX3]
CY = [CY1, CY2, CY3]
R = [R1, R2, R3]
