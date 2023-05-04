import jax.numpy as jnp
import numpy as np
from jax import jit, jacfwd, jacrev

#! Circular Obstacle Avoidance


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
obstacle_avoidance_bf_1 = lambda x: h(x, CX1, CY1, R1)
obstacle_avoidance_bj_1 = lambda x: dhdx(x, CX1, CY1, R1)
obstacle_avoidance_bh_1 = lambda x: d2hdx2(x, CX1, CY1, R1)
CX2, CY2, R2 = 2.0, 2.0, 0.5
obstacle_avoidance_bf_2 = lambda x: h(x, CX2, CY2, R2)
obstacle_avoidance_bj_2 = lambda x: dhdx(x, CX2, CY2, R2)
obstacle_avoidance_bh_2 = lambda x: d2hdx2(x, CX2, CY2, R2)
CX3, CY3, R3 = 0.0, 3.0, 0.5
obstacle_avoidance_bf_3 = lambda x: h(x, CX3, CY3, R3)
obstacle_avoidance_bj_3 = lambda x: dhdx(x, CX3, CY3, R3)
obstacle_avoidance_bh_3 = lambda x: d2hdx2(x, CX3, CY3, R3)


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

CX = [CX1, CX2, CX3]
CY = [CY1, CY2, CY3]
R = [R1, R2, R3]
