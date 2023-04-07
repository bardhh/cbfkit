from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt
import control as control
import cvxopt as cvxopt
from cbflib import cbf, tut_cbf_utils, tut_sys_and_ctrl
import matplotlib.animation as animation
from sympy import symbols, Matrix, sin, cos, lambdify, exp, sqrt, log, diff, Mul, srepr, solve

def cbf_find_param(x_0,x_goal,h,B,t0,t1):

    xr0, xr1, t, k = symbols(
    'xr0 xr1 t k')

    pre = B.subs([(t,0),(xr0,x_0[0]),(xr1,x_0[1])])
    return solve(pre, k)[0]
