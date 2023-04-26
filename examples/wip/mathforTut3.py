import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sympy import symbols, Matrix, sin, cos
import matplotlib.animation as animation
from cbflib import cbf, tut_cbf_utils, tut_sys_and_ctrl
from sympy import (
    symbols,
    Matrix,
    sin,
    cos,
    lambdify,
    exp,
    sqrt,
    log,
    diff,
    Mul,
    srepr,
    Symbol,
)

# import conditional if system is mac m1
if platform.system() == "Darwin" and platform.machine() == "arm64":
    from kvxopt import matrix, solvers
else:
    from cvxopt import matrix, solvers

# Robot Goal
x_goal = np.array([5, 5])

# Undesired areas in ellipse format (x,y,rad_x,rad_y) - Use example(0) through example(3)
bad_sets = tut_cbf_utils.example(3)

# Parameters for reference controller
ctrl_param = [10]

# Symbols and equations for the CBF
xr0, xr1, xr2, cx, cy, rad_x, rad_y, xr2_dot, u = symbols(
    "xr0 xr1 xr2 cx cy rad_x rad_y xr2_dot u"
)
symbs = (cx, cy, rad_x, rad_y, xr0, xr1, xr2)

# Barrier function - distance of robot to obstacle
B = ((xr0 - cx) / rad_x) ** 2 + ((xr1 - cy) / rad_y) ** 2 - 1


# dx = f(x) + g(x)u
f = Matrix([cos(xr2), sin(xr2), 0])
g = Matrix([0, 0, 1])
states_dot = Matrix([cos(xr2), sin(xr2), xr2_dot])

dB_dxr0 = diff(B, xr0)
dB_dxr1 = diff(B, xr1)
dB_dxr2 = diff(B, xr2)

L_fBx = np.matmul(np.array([dB_dxr0, dB_dxr1]).T, np.array([cos(xr2), sin(xr2)]))

L_gBx = dB_dxr2 * xr2_dot


dL_fBxr0 = diff(L_fBx, xr0)
dL_fBxr1 = diff(L_fBx, xr1)

L2_fBx = np.matmul(np.array([dL_fBxr0, dL_fBxr1]).T, np.array([cos(xr2), sin(xr2)]))

dL_fBxr2 = diff(L_fBx, xr2)

LgLf_fBx = dL_fBxr2 * xr2_dot

a_2 = 10
a_1 = 2

rest = a_2 * (L_fBx + a_1 * B)

getall = L2_fBx + LgLf_fBx + rest + L_fBx

g1 = getall.subs(cy, 2)
g2 = g1.subs(cx, 2)
g3 = g2.subs(rad_x, 1)
g4 = g3.subs(rad_y, 1)
g5 = g4.subs([(xr0, 0.98), (xr1, 0.98), (xr2, 0), (xr2_dot, 1)])
float(g5)
